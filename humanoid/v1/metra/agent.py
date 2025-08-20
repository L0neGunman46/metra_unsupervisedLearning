import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from .utils import _to_t
from .networks import Representation, Actor, Critic


class METRAAgent:
    def __init__(self, state_dim, action_dim, config):
        self.config = config
        self.device = config["device"]
        self.skill_dim = config["skill_dim"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.discrete_skills = config.get("discrete_skills", False)
        
        hidden_dim = config["hidden_dim"]
        self.phi = Representation(state_dim, self.skill_dim, hidden_dim).to(self.device)
        self.actor = Actor(state_dim, action_dim,self.skill_dim, hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.skill_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim,action_dim, self.skill_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        phi_lr = config.get("phi_lr", config["lr"])
        dual_lam_lr = config.get("dual_lam_lr", config["lr"])
        
        
        self.phi_optimizer = optim.Adam(self.phi.parameters(), lr=phi_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config["lr"])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config["lr"])
        self.use_dual = bool(config.get("dual_reg", True))
        if self.use_dual:
            lam_init = float(config.get("dual_lam_init", 30.0))
            self.log_dual_lam = torch.tensor(
                np.log(lam_init), requires_grad=True, device=self.device
            )
            self.dual_lam_optimizer = optim.Adam([self.log_dual_lam], lr=dual_lam_lr)
            self.dual_slack = float(config.get("dual_slack", 1e-3))
            self.dual_lam_max = float(config.get("dual_lam_max", 1e3))
        
        self.log_alpha= torch.tensor(
            np.log(config["alpha"]),requires_grad=True, device=self.device
        )
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config["lr"])
        self.target_entropy = -float(action_dim)
    
    
    def select_action(self,state, skill):
        s = _to_t(state, self.device).unsqueeze(0)
        z = _to_t(skill, self.device).unsqueeze(0)
        action, _ = self.actor.sample(s, z)
        return action.detach().cpu().numpy().squeeze(0)

   
    def update(self, replay_buffer):
        log_info = collections.defaultdict(list)

        for _ in range(self.config["trans_optimization_epochs"]):
            # Sample batch from replay buffer
            state, action, _, next_state, done, skill = replay_buffer.sample(
                self.config["batch_size"]
            )

            s = _to_t(state, self.device)
            a = _to_t(action, self.device)
            sp = _to_t(next_state, self.device)
            d = _to_t(done, self.device)
            z = _to_t(skill, self.device)

            # 1) Representation learning and dual constraint
            phi_s = self.phi(s)
            with torch.no_grad():
                phi_sp = self.phi(sp)

            # Intrinsic reward: (φ(s') - φ(s))^T z
            delta_phi = phi_sp - phi_s
            intrinsic_reward = torch.sum(delta_phi * z, dim=1, keepdim=True)

            # Temporal-Lipschitz constraint
            eps = self.dual_slack
            lip_sq = torch.sum(delta_phi**2, dim=1)
            c = torch.minimum(torch.full_like(lip_sq, eps), 1.0 - lip_sq)

            if self.use_dual:
                dual_lam = self.log_dual_lam.exp()
                phi_obj = intrinsic_reward.squeeze(1) + dual_lam.detach() * c
                phi_loss = -phi_obj.mean()
            else:
                phi_loss = -intrinsic_reward.mean()

            self.phi_optimizer.zero_grad()
            phi_loss.backward()
            nn.utils.clip_grad_norm_(self.phi.parameters(),
                                 max_norm=self.config.get("phi_grad_norm_clip", 10.0))
            self.phi_optimizer.step()

            if self.use_dual:
                # Dual variable update
                lambda_loss = (dual_lam * c.detach()).mean()
                self.dual_lam_optimizer.zero_grad()
                lambda_loss.backward()
                nn.utils.clip_grad_norm_([self.log_dual_lam],
                                     max_norm=self.config.get("dual_lam_grad_norm_clip", 1.0))
                self.dual_lam_optimizer.step()

                # Cap λ to prevent runaway growth
                with torch.no_grad():
                    if self.log_dual_lam.exp().item() > self.dual_lam_max:
                        self.log_dual_lam.copy_(
                        torch.log(torch.tensor(self.dual_lam_max, device=self.device))
                        )

                log_info["lambda_loss"].append(lambda_loss.item())
                log_info["dual_lambda"].append(dual_lam.item())
                violation = torch.relu(lip_sq - 1.0)
                log_info["constraint_violation"].append(violation.mean().item())

            # 2) Critic update (SAC)
            r = intrinsic_reward.detach()
            with torch.no_grad():
                next_action, next_logp = self.actor.sample(sp, z)
                tq1, tq2 = self.critic_target(sp, next_action, z)
                alpha = self.log_alpha.exp()
                target_q = torch.min(tq1, tq2) - alpha * next_logp
                target_q = r + (1.0 - d) * self.gamma * target_q

            q1, q2 = self.critic(s, a, z)
            critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                 max_norm=self.config.get("critic_grad_norm_clip", 10.0))
            self.critic_optimizer.step()

            # 3) Actor update (SAC)
            pi, log_pi = self.actor.sample(s, z)
            q1_pi, q2_pi = self.critic(s, pi, z)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha = self.log_alpha.exp()
            actor_loss = (alpha.detach() * log_pi - min_q_pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                                    max_norm=self.config.get("actor_grad_norm_clip", 10.0))
            self.actor_optimizer.step()

            # 4) Temperature α update
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha],
                                 max_norm=self.config.get("alpha_grad_norm_clip", 1.0))
            self.alpha_optimizer.step()

            # 5) Soft update target critic
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

        # Logging
            log_info["phi_loss"].append(phi_loss.item())
            log_info["critic_loss"].append(critic_loss.item())
            log_info["actor_loss"].append(actor_loss.item())
            log_info["alpha_loss"].append(alpha_loss.item())
            log_info["alpha"].append(self.log_alpha.exp().item())
            log_info["intrinsic_reward"].append(intrinsic_reward.mean().item())
            log_info["q_value"].append(q1.mean().item())

        # Average logs over all optimization epochs
        final_logs = {k: float(np.mean(v)) for k, v in log_info.items()}
        return final_logs

    def sample_skill(self):
        if self.discrete_skills:
            idx = np.random.randint(self.skill_dim)
            z = np.zeros(self.skill_dim, dtype=np.float32)
            z[idx] = 1.0
            z = z - np.mean(z)
        else:
            z = np.random.randn(self.skill_dim).astype(np.float32)
            if self.config["unit_length_skill"]:
                z = z / (np.linalg.norm(z) + 1e-8)
        return z

    def select_skill_for_goal(self, current_state, goal_state):
        with torch.no_grad():
            s = _to_t(current_state, self.device).unsqueeze(0)
            g = _to_t(goal_state, self.device).unsqueeze(0)
            dvec = (self.phi(g) - self.phi(s)).cpu().numpy().squeeze(0)
        if self.discrete_skills:
            z = np.zeros(self.skill_dim, dtype=np.float32)
            z[int(np.argmax(dvec))] = 1.0
            z = z - np.mean(z)
        else:
            z = dvec.astype(np.float32)
            z = z / (np.linalg.norm(z) + 1e-8)
        return z
    
    def save(self, filename):
        torch.save(
            {
                "phi": self.phi.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "log_dual_lam": self.log_dual_lam.detach().cpu()
                if self.use_dual
                else None,
            },
            filename,
        )
    
    def load(self, filename):
        ckpt = torch.load(filename, map_location=self.device)
        self.phi.load_state_dict(ckpt["phi"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_()
        if ckpt.get("log_dual_lam") is not None:
            self.log_dual_lam = ckpt["log_dual_lam"].to(self.device).requires_grad_()
            self.use_dual = True