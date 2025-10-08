# 🌐 Remote Deployment Guide - LW-FedSSL

> **Complete guide to deploying federated learning with remote clients using ngrok**

This guide shows you how to run your **LW-FedSSL server** on your local machine and connect **remote clients** from anywhere in the world (Google Colab, Kaggle, AWS, etc.) using **ngrok TCP tunneling**.

---

## 📋 What You'll Accomplish

- ✅ Run server on your local machine (MacBook, PC, etc.)
- ✅ Expose server to internet using ngrok
- ✅ Connect clients from Google Colab notebooks
- ✅ Connect clients from Kaggle kernels
- ✅ Connect clients from remote servers (AWS, Azure, etc.)
- ✅ Monitor training in real-time
- ✅ Handle multiple remote clients simultaneously

---

## 🎯 Architecture

```
Your Local Machine               Internet                Remote Clients
┌──────────────┐                                     ┌─────────────┐
│              │                                     │ Colab GPU 1 │
│  FL Server   │◄─────┐                        ┌────┤ Client 1    │
│  (port 8080) │      │                        │    └─────────────┘
└──────────────┘      │      ┌───────────┐    │
                      ├─────►│   ngrok   │◄───┤    ┌─────────────┐
┌──────────────┐      │      │  Tunnel   │    │    │ Kaggle TPU  │
│    ngrok     │──────┘      └───────────┘    ├────┤ Client 2    │
│  (port 4040) │                               │    └─────────────┘
└──────────────┘                               │
                                               │    ┌─────────────┐
                                               │    │ AWS EC2     │
                                               └────┤ Client 3    │
                                                    └─────────────┘
```

---

## 🚀 Complete Step-by-Step Guide

### Part 1: Server Setup (Your Local Machine)

#### Step 1.1: Install ngrok

**macOS:**
```bash
brew install ngrok
```

**Linux:**
```bash
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xvzf ngrok-v3-stable-linux-amd64.tgz
sudo mv ngrok /usr/local/bin
```

**Windows:**
```powershell
# Download from https://ngrok.com/download
# Extract and add to PATH
```

**Verify installation:**
```bash
ngrok version
# Should output: ngrok version 3.x.x
```

#### Step 1.2: Setup ngrok Account (Free)

1. Visit [https://ngrok.com/signup](https://ngrok.com/signup)
2. Create free account (no credit card required)
3. Copy your auth token from dashboard
4. Configure ngrok:

```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN_HERE
```

#### Step 1.3: Prepare Your Server

```bash
# Navigate to project
cd LW_FedSSL

# Verify dependencies
pip install -r requirements.txt

# Check configuration (optional)
cat config.py | grep SERVER
```

Should show:
```python
SERVER_ADDRESS: str = "0.0.0.0"  # Binds to all interfaces
SERVER_PORT: int = 8080
```

#### Step 1.4: Start Server

**Terminal 1** - Run the server:
```bash
python3 enhanced_server.py --mode lwfedssl
```

You should see:
```
🚀 Starting LW-FedSSL Server
📊 Mode: lwfedssl
🌐 Address: 0.0.0.0:8080
⏳ Waiting for clients...
```

**Keep this terminal running!**

#### Step 1.5: Create ngrok Tunnel

**Terminal 2** - Open new terminal and run:
```bash
ngrok tcp 8080
```

You'll see output like:
```
ngrok                                                                                                      

Session Status    online                                                                                   
Account           your-email@example.com (Plan: Free)                                                      
Version           3.5.0                                                                                    
Region            United States (us)                                                                       
Latency           -                                                                                        
Web Interface     http://127.0.0.1:4040                                                                    
Forwarding        tcp://0.tcp.in.ngrok.io:17027 -> localhost:8080                                          

Connections       ttl     opn     rt1     rt5     p50     p90                                              
                  0       0       0.00    0.00    0.00    0.00
```

**⭐ COPY THIS ADDRESS:** `0.tcp.in.ngrok.io:17027`

This is your **public server address** that remote clients will use!

**Keep this terminal running too!**

---

### Part 2: Client Setup (Remote Machines)

You now have **3 options** for remote clients:

---

#### Option A: Google Colab Client

**Perfect for:** Free GPU training, easy demos

1. **Open Google Colab:** [https://colab.research.google.com](https://colab.research.google.com)

2. **Create new notebook**

3. **Copy-paste this complete setup:**

```python
# Cell 1: Install dependencies
!pip install -q flwr==1.5.0 torch torchvision

# Cell 2: Clone repository
!git clone https://github.com/YOUR_USERNAME/LW_FedSSL.git
%cd LW_FedSSL

# Cell 3: Check GPU
import torch
print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Connect to server - REPLACE WITH YOUR NGROK ADDRESS!
SERVER_ADDRESS = "0.tcp.in.ngrok.io:17027"  # ← Change this!
CLIENT_ID = 1  # Change for each client (1, 2, 3, etc.)

# Cell 5: Run all 3 stages
for stage in [1, 2, 3]:
    print(f"\n{'='*60}")
    print(f"🚀 Starting Stage {stage}")
    print(f"{'='*60}\n")
    
    !python3 client_app.py \
        --server {SERVER_ADDRESS} \
        --client-id {CLIENT_ID} \
        --stage {stage}
```

4. **Run all cells** (Runtime → Run all)

5. **Watch training happen!**

**For multiple Colab clients:**
- Open 2-3 different notebooks
- Change `CLIENT_ID = 2`, `CLIENT_ID = 3`, etc.
- Run simultaneously

---

#### Option B: Kaggle Kernel Client

**Perfect for:** TPU training, public datasets

1. **Go to Kaggle:** [https://www.kaggle.com](https://www.kaggle.com)

2. **Create new notebook** (Code → New Notebook)

3. **Settings:**
   - Enable GPU/TPU: Settings → Accelerator → GPU
   - Internet: Settings → Internet → On

4. **Add code cells:**

```python
# Cell 1: Install Flower
!pip install -q flwr==1.5.0

# Cell 2: Clone your repository
!git clone https://github.com/YOUR_USERNAME/LW_FedSSL.git
import os
os.chdir('LW_FedSSL')

# Cell 3: Run client
SERVER = "0.tcp.in.ngrok.io:17027"  # Your ngrok address
CLIENT_ID = 2  # Different ID for each kernel

for stage in [1, 2, 3]:
    !python3 client_app.py --server {SERVER} --client-id {CLIENT_ID} --stage {stage}
```

---

#### Option C: AWS/Azure/Remote Server

**Perfect for:** Production deployment, scalability

**SSH into your remote machine:**
```bash
ssh user@remote-server.com
```

**Setup (one-time):**
```bash
# Install Python and dependencies
sudo apt update
sudo apt install -y python3 python3-pip git

# Clone repository
git clone https://github.com/YOUR_USERNAME/LW_FedSSL.git
cd LW_FedSSL

# Install requirements
pip3 install -r requirements.txt
```

**Run client:**
```bash
# Stage 1
python3 client_app.py \
    --server 0.tcp.in.ngrok.io:17027 \
    --client-id 3 \
    --stage 1

# Stage 2
python3 client_app.py \
    --server 0.tcp.in.ngrok.io:17027 \
    --client-id 3 \
    --stage 2

# Stage 3
python3 client_app.py \
    --server 0.tcp.in.ngrok.io:17027 \
    --client-id 3 \
    --stage 3
```

**Run in background with nohup:**
```bash
nohup python3 client_app.py --server 0.tcp.in.ngrok.io:17027 --client-id 3 --stage 1 > client.log 2>&1 &

# Check logs
tail -f client.log
```

---

### Part 3: Advanced Client Setup (Enhanced Features)

If you want to use the **enhanced client** with more features (non-IID data, metrics, etc.):

```bash
# On remote machine
python3 enhanced_client.py \
    --server 0.tcp.in.ngrok.io:17027 \
    --client-id 5 \
    --mode lwfedssl \
    --data-distribution non_iid_dirichlet \
    --alpha 0.5
```

**Enhanced client options:**
- `--data-distribution`: `iid`, `non_iid_label`, `non_iid_dirichlet`
- `--alpha`: Dirichlet concentration (0.1 = very non-IID, 1.0 = more IID)
- `--mode`: `lwfedssl` or `baseline`

---

## 📊 Monitoring Your Training

### On Your Server Machine (Terminal 1)

You'll see real-time updates:
```
✅ Client 1 connected from 34.123.45.67
✅ Client 2 connected from 52.234.56.78
✅ Client 3 connected from 18.234.67.89

📊 Round 1/10
  → Selecting clients...
  → Selected: [1, 2, 3]
  → Sending parameters...
  → Collecting updates...
  → Aggregating...
  ✅ Loss: 4.73, Communication: 0.5 MB

📊 Round 2/10
  → Loss: 4.21, Communication: 0.5 MB
...
```

### ngrok Web Interface

Open your browser: **http://localhost:4040**

You'll see:
- 🌐 Real-time connection logs
- 📊 Request/response details
- 🕐 Connection timestamps
- 🌍 Client IP addresses and locations

### Check Logs

```bash
# View system logs
tail -f logs/<session_id>/system.jsonl

# View metrics
tail -f logs/<session_id>/metrics.jsonl

# View client activity
tail -f logs/<session_id>/client_activity.jsonl
```

---

## 🐛 Troubleshooting

### Issue: "Connection refused"

**Check server is running:**
```bash
ps aux | grep enhanced_server
```

**Check ngrok is active:**
```bash
# Should see "Forwarding" line in Terminal 2
# Or visit http://localhost:4040
```

**Verify address format:**
- ✅ Correct: `0.tcp.in.ngrok.io:17027`
- ❌ Wrong: `http://0.tcp.in.ngrok.io:17027`
- ❌ Wrong: `0.tcp.in.ngrok.io` (missing port)

---

### Issue: "Connection timeout"

**Increase timeout in `config.py`:**
```python
# Add to SystemConfig class
SERVER_TIMEOUT: int = 600  # 10 minutes
```

**Or in server code:**
```python
config = ServerConfig(
    num_rounds=10,
    round_timeout=600.0  # 10 minutes
)
```

---

### Issue: Firewall blocking connections

**macOS:**
```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/python3
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblock /usr/local/bin/python3
```

**Linux:**
```bash
sudo ufw allow 8080/tcp
```

**Windows:**
```powershell
New-NetFirewallRule -DisplayName "Python FL Server" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow
```

---

### Issue: ngrok session expired

**Free ngrok sessions expire after 2 hours**

**Solutions:**
1. **Restart ngrok** - Get new address, update clients
2. **Upgrade to paid plan** - Get persistent domains
3. **Use reserved domains** (paid feature):
```bash
ngrok tcp 8080 --subdomain=your-reserved-name
# Always get: your-reserved-name.ngrok.io
```

---

### Issue: "Address already in use"

**Kill existing process on port 8080:**
```bash
# Find process
lsof -ti:8080

# Kill it
lsof -ti:8080 | xargs kill -9

# Restart server
python3 enhanced_server.py --mode lwfedssl
```

---

## 🔒 Security Best Practices

### For Testing/Demo:
✅ **Current setup is fine** - ngrok provides HTTPS encryption

### For Production:

1. **Add authentication to ngrok:**
```bash
ngrok tcp 8080 --auth="username:password"
```

Clients must use:
```bash
# Not supported by basic FL client - need custom implementation
```

2. **Use private ngrok domains** (paid):
```bash
ngrok tcp 8080 --subdomain=my-private-server
```

3. **Implement client authentication:**

Add to `client_app.py`:
```python
# Add API key to client
CLIENT_API_KEY = "your-secret-key"

# Include in fit() config
config = {
    "stage": stage,
    "api_key": CLIENT_API_KEY
}
```

Add to `enhanced_server.py`:
```python
def configure_fit(self, server_round, parameters, client_manager):
    # Verify API key
    if config.get("api_key") != EXPECTED_KEY:
        raise ValueError("Invalid API key")
```

4. **Use SSL/TLS certificates:**

Configure Flower with certificates:
```python
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=config,
    strategy=strategy,
    certificates=(
        Path("server_cert.pem").read_bytes(),
        Path("server_key.pem").read_bytes(),
    ),
)
```

5. **Monitor ngrok dashboard:**
- Visit http://localhost:4040
- Check for suspicious IPs
- Monitor request patterns

---

## 📈 Scaling to Many Clients

### Running 10+ Remote Clients

**1. Increase server capacity:**

```python
# In config.py
MIN_CLIENTS_PER_ROUND: int = 5  # Lower minimum
MAX_CLIENTS_PER_ROUND: int = 20  # Higher maximum

# In enhanced_server.py
config = ServerConfig(
    num_rounds=10,
    round_timeout=600.0  # Longer timeout for many clients
)
```

**2. Stagger client startup:**

Instead of starting all clients simultaneously:

```bash
# Client 1
python3 client_app.py --server <ngrok> --client-id 1 --stage 1 &
sleep 5

# Client 2
python3 client_app.py --server <ngrok> --client-id 2 --stage 1 &
sleep 5

# Client 3
python3 client_app.py --server <ngrok> --client-id 3 --stage 1 &
```

**3. Use orchestration script:**

Create `start_multiple_clients.sh`:
```bash
#!/bin/bash
SERVER="0.tcp.in.ngrok.io:17027"
NUM_CLIENTS=10

for i in $(seq 1 $NUM_CLIENTS); do
    echo "Starting client $i..."
    python3 client_app.py --server $SERVER --client-id $i --stage 1 &
    sleep 3
done

wait
echo "All clients finished!"
```

Run:
```bash
chmod +x start_multiple_clients.sh
./start_multiple_clients.sh
```

---

## 🎯 Complete Example: 5 Remote Clients

### Your MacBook (Server)

**Terminal 1:**
```bash
cd LW_FedSSL
python3 enhanced_server.py --mode lwfedssl
```

**Terminal 2:**
```bash
ngrok tcp 8080
# Copy address: 0.tcp.in.ngrok.io:17027
```

### Colab Notebook 1
```python
!pip install -q flwr==1.5.0 torch torchvision
!git clone https://github.com/YOUR_USERNAME/LW_FedSSL.git
%cd LW_FedSSL

for stage in [1,2,3]:
    !python3 client_app.py --server 0.tcp.in.ngrok.io:17027 --client-id 1 --stage {stage}
```

### Colab Notebook 2
```python
# Same as above but:
# --client-id 2
```

### Kaggle Kernel 1
```python
# Same as above but:
# --client-id 3
```

### AWS EC2 Instance
```bash
ssh ubuntu@ec2-instance.amazonaws.com
cd LW_FedSSL
python3 client_app.py --server 0.tcp.in.ngrok.io:17027 --client-id 4 --stage 1
# ... stages 2, 3
```

### Your Friend's Laptop
```bash
cd LW_FedSSL
python3 client_app.py --server 0.tcp.in.ngrok.io:17027 --client-id 5 --stage 1
# ... stages 2, 3
```

**Result:** 5 clients training simultaneously from 5 different locations! 🎉

---

## 📚 Resources

- **ngrok Documentation:** https://ngrok.com/docs
- **Flower Framework:** https://flower.dev/docs
- **Google Colab:** https://colab.research.google.com
- **Kaggle Kernels:** https://www.kaggle.com/code

---

## 🤝 Support

**Having issues?**
1. Check the [Troubleshooting](#-troubleshooting) section above
2. Visit ngrok dashboard: http://localhost:4040
3. Check server logs: `logs/<session>/system.jsonl`
4. Open an issue on GitHub

---

## ✅ Checklist for Remote Deployment

Before starting:
- [ ] Server machine has stable internet
- [ ] ngrok is installed and authenticated
- [ ] Repository is on GitHub (for easy client cloning)
- [ ] `requirements.txt` is up-to-date
- [ ] Tested locally with `simulate_clients.py`

During deployment:
- [ ] Server running (Terminal 1)
- [ ] ngrok tunnel active (Terminal 2)
- [ ] Copied ngrok address
- [ ] Clients can access GitHub
- [ ] Client IDs are unique
- [ ] Firewall allows connections

After training:
- [ ] Check logs: `logs/<session>/`
- [ ] Generate visualizations: `python3 visualize.py`
- [ ] Save results: `checkpoints/lwfedssl_final.pt`
- [ ] Stop ngrok (Ctrl+C)
- [ ] Stop server (Ctrl+C)

---

**🎉 You're now ready to deploy federated learning across the internet!**

Happy training! 🚀
