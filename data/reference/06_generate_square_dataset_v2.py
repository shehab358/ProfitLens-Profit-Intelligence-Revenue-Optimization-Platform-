"""Synthetic Square-POS-style retail dataset v2.
24 months (Jan 2024 - Dec 2025), 30 stores, full US retail event calendar."""
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

rng = np.random.default_rng(42)

# ---------------- Stores ----------------
cities = ["Austin","Dallas","Houston","San Antonio","Phoenix","Denver","Seattle","Portland",
          "Chicago","Detroit","Columbus","Atlanta","Miami","Orlando","Tampa","Charlotte",
          "Nashville","Memphis","Boston","New York","Brooklyn","Philadelphia","Baltimore",
          "Las Vegas","San Diego","Los Angeles","San Jose","Sacramento","Minneapolis","Kansas City"]
stores = [f"Store {i+1:02d} - {c}" for i, c in enumerate(cities)]
tiers = rng.choice([1.8, 1.0, 0.55], size=30, p=[0.2, 0.5, 0.3])
store_tier = dict(zip(stores, tiers))

# ---------------- Catalog ----------------
catalog = [
    ("Apparel","Classic Denim Jacket",{"S":79.00,"M":79.00,"L":79.00,"XL":84.00},0.45,0.9),
    ("Apparel","Cotton Crew T-Shirt",{"S":19.50,"M":19.50,"L":19.50,"XL":21.00},0.35,2.5),
    ("Apparel","Slim Fit Chinos",{"30x30":54.00,"32x32":54.00,"34x32":54.00},0.42,1.2),
    ("Apparel","Hooded Sweatshirt",{"S":45.00,"M":45.00,"L":45.00,"XL":48.00},0.40,1.6),
    ("Apparel","Flannel Shirt",{"S":39.50,"M":39.50,"L":39.50},0.38,1.0),
    ("Apparel","Summer Linen Dress",{"S":68.00,"M":68.00,"L":68.00},0.44,0.8),
    ("Apparel","Performance Leggings",{"S":42.00,"M":42.00,"L":42.00},0.36,1.4),
    ("Footwear","Canvas Sneakers",{"8":59.00,"9":59.00,"10":59.00,"11":59.00},0.48,1.5),
    ("Footwear","Leather Boots",{"8":129.00,"9":129.00,"10":129.00,"11":129.00},0.52,0.6),
    ("Footwear","Running Shoes",{"8":89.00,"9":89.00,"10":89.00,"11":89.00},0.50,1.1),
    ("Footwear","Slide Sandals",{"8":24.00,"9":24.00,"10":24.00},0.33,1.3),
    ("Accessories","Leather Wallet",{"Regular":40.00},0.38,1.4),
    ("Accessories","Canvas Tote Bag",{"Regular":30.00},0.30,1.7),
    ("Accessories","Baseball Cap",{"Regular":22.00},0.32,1.8),
    ("Accessories","Wool Scarf",{"Regular":28.00},0.35,0.7),
    ("Accessories","Polarized Sunglasses",{"Regular":65.00},0.28,1.0),
    ("Accessories","Leather Belt",{"32":35.00,"34":35.00,"36":35.00},0.36,1.1),
    ("Electronics","Wireless Earbuds",{"Black":79.00,"White":79.00},0.55,1.3),
    ("Electronics","Bluetooth Speaker",{"Regular":59.00},0.53,0.9),
    ("Electronics","Phone Charging Cable",{"1m":14.00,"2m":18.00},0.30,2.2),
    ("Electronics","Power Bank 10000mAh",{"Regular":34.00},0.50,1.2),
    ("Electronics","Smart Watch Band",{"S/M":25.00,"M/L":25.00},0.35,0.9),
    ("Home Goods","Scented Soy Candle",{"8oz":18.00,"12oz":26.00},0.30,1.9),
    ("Home Goods","Ceramic Coffee Mug",{"Regular":16.00},0.28,2.0),
    ("Home Goods","Throw Blanket",{"Regular":49.00},0.42,0.8),
    ("Home Goods","Picture Frame 8x10",{"Regular":22.00},0.34,1.0),
    ("Home Goods","Stainless Water Bottle",{"20oz":28.00,"32oz":34.00},0.35,1.6),
    ("Beauty","Hand Cream",{"Regular":12.00},0.30,2.1),
    ("Beauty","Facial Moisturizer",{"Regular":32.00},0.32,1.2),
    ("Beauty","Lip Balm Set",{"Regular":9.50},0.28,2.4),
    ("Beauty","Perfume Roll-On",{"Regular":24.00},0.30,1.0),
    ("Beauty","Bath Bomb Trio",{"Regular":15.00},0.29,1.5),
]
sku = 1000; items = []
for cat, name, variations, cost_r, pop in catalog:
    for var, price in variations.items():
        sku += 1
        items.append({"Category":cat,"Item":name,"Price Point Name":var,"SKU":f"SQ-{sku}",
                      "Unit Price":price,"Unit Cost":round(price*cost_r,2),"pop":pop})
items_df = pd.DataFrame(items)
base_pop = items_df["pop"].to_numpy()
cats = items_df["Category"].to_numpy()

# ---------------- Retail event calendar ----------------
def nth_weekday(year, month, weekday, n):
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n-1)

def last_weekday(year, month, weekday):
    d = date(year, month, 28) + timedelta(days=4)
    d -= timedelta(days=d.day)  # last day of month
    while d.weekday() != weekday: d -= timedelta(days=1)
    return d

EASTER = {2024: date(2024,3,31), 2025: date(2025,4,20)}

def build_calendar(years):
    """Return dict date -> event spec: traffic mult, category skews, discount profile, refund mult, closed."""
    cal = {}
    def add(d, name, traffic=1.0, cat_skew=None, disc=None, refund=1.0, closed=False):
        # keep the strongest event if a day collides
        if d in cal and cal[d]["traffic"] >= traffic and not closed: return
        cal[d] = {"event":name,"traffic":traffic,"cat_skew":cat_skew or {},
                  "disc":disc,"refund":refund,"closed":closed}
    for y in years:
        thx = nth_weekday(y,11,3,4)            # 4th Thursday Nov
        bf  = thx + timedelta(days=1)
        cm  = thx + timedelta(days=4)
        ss  = last_weekday(y,12,5)             # last Saturday of Dec
        if ss >= date(y,12,25): ss -= timedelta(days=7)  # last Sat BEFORE Christmas
        add(date(y,1,1),  "New Year's Day", 0.5)
        add(date(y,2,13), "Valentine's Eve", 1.35, {"Beauty":1.8,"Accessories":1.6})
        add(date(y,2,14), "Valentine's Day", 1.55, {"Beauty":2.0,"Accessories":1.7})
        add(EASTER[y],    "Easter (mostly closed)", 0.3)
        md = nth_weekday(y,5,6,2)              # Mother's Day: 2nd Sunday May
        add(md - timedelta(days=1), "Mother's Day Eve", 1.35, {"Beauty":1.7,"Home Goods":1.5,"Accessories":1.4})
        add(md, "Mother's Day", 1.40, {"Beauty":1.8,"Home Goods":1.5,"Accessories":1.5})
        add(last_weekday(y,5,0), "Memorial Day", 1.35, disc=(0.45,[0.10,0.15,0.20,0.25]))
        fd = nth_weekday(y,6,6,3)              # Father's Day: 3rd Sunday June
        add(fd - timedelta(days=1), "Father's Day Eve", 1.25, {"Electronics":1.6,"Accessories":1.4})
        add(fd, "Father's Day", 1.30, {"Electronics":1.7,"Accessories":1.4})
        add(date(y,7,4), "Independence Day", 1.35, disc=(0.45,[0.10,0.15,0.20,0.25]))
        add(nth_weekday(y,9,0,1), "Labor Day", 1.35, disc=(0.45,[0.10,0.15,0.20,0.25]))
        add(thx, "Thanksgiving (mostly closed)", 0.15)
        add(bf,  "Black Friday", 2.8, {"Electronics":2.0,"Footwear":1.3},
            disc=(0.65,[0.20,0.25,0.30,0.40]))
        add(bf + timedelta(days=1), "BF Weekend Sat", 1.7, {"Electronics":1.5},
            disc=(0.55,[0.15,0.20,0.25,0.30]))
        add(bf + timedelta(days=2), "BF Weekend Sun", 1.5, {"Electronics":1.4},
            disc=(0.50,[0.15,0.20,0.25,0.30]))
        add(cm,  "Cyber Monday", 1.35, {"Electronics":2.2},
            disc=(0.60,[0.20,0.25,0.30,0.40]))
        add(ss,  "Super Saturday", 2.3, None, disc=(0.50,[0.15,0.20,0.25,0.30]))
        add(date(y,12,24), "Christmas Eve", 1.6)
        add(date(y,12,25), "Christmas Day (closed)", 0.0, closed=True)
        for k in range(6):                      # post-Christmas returns week
            add(date(y,12,26)+timedelta(days=k), "Post-Christmas Returns", 1.35, None,
                disc=(0.50,[0.20,0.25,0.30,0.40]), refund=3.5)
        add(date(y,12,31), "New Year's Eve", 1.2)
        for k in range(5):                      # early-Jan return tail
            add(date(y,1,2)+timedelta(days=k), "January Returns Tail", 0.85, refund=2.5)
    return cal

def month_mult(d):
    m = d.month
    if m == 12: return 1.15 + 0.45*min(d.day,23)/23.0   # December ramp to ~1.6
    return {1:0.80, 2:0.92, 3:0.98, 4:1.00, 5:1.05, 6:1.02, 7:1.00,
            8:1.12, 9:0.98, 10:1.02, 11:1.10}[m]

BTS_SKEW = {"Apparel":1.4,"Footwear":1.35}               # back-to-school Aug

# ---------------- Simulation ----------------
start = date(2024,1,1); end = date(2025,12,31)
years = [2024, 2025]
cal = build_calendar(years)
n_days = (end-start).days + 1
payment_p = {"Card":0.71,"Cash":0.24,"Other":0.05}
devices = ["Register 1","Register 2","iPad POS"]
BASE = 22
rows = []; cal_rows = []; txn_seq = 1000000

for di in range(n_days):
    d = start + timedelta(days=di)
    ev = cal.get(d, {"event":"","traffic":1.0,"cat_skew":{},"disc":None,"refund":1.0,"closed":False})
    dow_m = [0.9,0.85,0.9,1.0,1.25,1.55,1.35][d.weekday()]
    season = 1.0 + 0.08*np.sin(2*np.pi*(d.timetuple().tm_yday)/365.25)
    yoy = 1.06 if d.year == 2025 else 1.0
    mm = month_mult(d)
    traffic = dow_m * season * mm * ev["traffic"] * yoy
    # category weights of the day
    skew = dict(ev["cat_skew"])
    if d.month == 8:
        for c,v in BTS_SKEW.items(): skew[c] = max(skew.get(c,1.0), v)
    w = base_pop * np.array([skew.get(c,1.0) for c in cats])
    w = w / w.sum()
    # discount profile of the day
    if ev["disc"]:
        disc_p, disc_depths = ev["disc"]
    elif d.month == 12:
        disc_p, disc_depths = 0.40, [0.10,0.15,0.20,0.25]
    else:
        disc_p, disc_depths = 0.30, [0.10,0.15,0.20]
    refund_p = min(0.015 * ev["refund"], 0.06)
    cal_rows.append([d.isoformat(), d.strftime("%A"), ev["event"], round(traffic/ (dow_m*yoy),3),
                     round(dow_m,2), round(mm,3), round(ev["traffic"],2), disc_p,
                     round(refund_p,4), int(ev["closed"])])
    if ev["closed"]: continue
    for store in stores:
        n_txn = rng.poisson(BASE * store_tier[store] * traffic)
        for _ in range(n_txn):
            txn_seq += 1
            txn_id = f"TXN{txn_seq}"
            hour = int(np.clip(rng.normal(14.5,3.2),9,20)); minute = rng.integers(0,60)
            n_lines = rng.choice([1,2,3,4],p=[0.55,0.28,0.12,0.05])
            pay = rng.choice(list(payment_p), p=list(payment_p.values()))
            device = devices[rng.integers(0,3)]
            cust = f"CUST{rng.integers(1,20000):05d}" if rng.random()<0.35 else ""
            idx = rng.choice(len(items_df), size=n_lines, replace=False, p=w)
            for i in idx:
                it = items_df.iloc[i]
                qty = int(rng.choice([1,1,1,2,2,3],p=[0.55,0.15,0.10,0.12,0.05,0.03]))
                gross = round(it["Unit Price"]*qty,2)
                disc = round(gross*rng.choice(disc_depths),2) if rng.random()<disc_p else 0.0
                net = round(gross-disc,2); tax = round(net*0.0825,2)
                rows.append([d.isoformat(),f"{hour:02d}:{minute:02d}","CST",it["Category"],it["Item"],
                             qty,it["Price Point Name"],it["SKU"],it["Unit Price"],gross,
                             -disc if disc else 0.0,net,tax,round(net+tax,2),txn_id,pay,device,store,
                             cust,"Payment",it["Unit Cost"],round(net-it["Unit Cost"]*qty,2)])
            if rng.random() < refund_p:
                i = rng.choice(len(items_df), p=w); it = items_df.iloc[i]
                gross = round(it["Unit Price"],2); tax = round(gross*0.0825,2); txn_seq += 1
                rows.append([d.isoformat(),f"{hour:02d}:{minute:02d}","CST",it["Category"],it["Item"],
                             -1,it["Price Point Name"],it["SKU"],it["Unit Price"],-gross,0.0,-gross,
                             -tax,round(-(gross+tax),2),f"TXN{txn_seq}",pay,device,store,"","Refund",
                             it["Unit Cost"],round(-gross+it["Unit Cost"],2)])

cols = ["Date","Time","Time Zone","Category","Item","Qty","Price Point Name","SKU","Unit Price",
        "Gross Sales","Discounts","Net Sales","Tax","Total Collected","Transaction ID",
        "Payment Method","Device Name","Location","Customer ID","Event Type","Unit Cost","Gross Profit"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("/home/claude/v2/square_item_sales_detail_24mo.csv", index=False)

cal_df = pd.DataFrame(cal_rows, columns=["Date","Weekday","Event","Season x Month x Event Mult",
        "DOW Mult","Month Mult","Event Mult","Discount Line Prob","Refund Txn Prob","Store Closed"])
cal_df.to_csv("/home/claude/v2/ground_truth_event_calendar.csv", index=False)

# aggregates
agg = (df.groupby(["Location","Category","Item","SKU","Price Point Name","Unit Price"], as_index=False)
         .agg(**{"Items Sold":("Qty","sum"),"Gross Sales":("Gross Sales","sum"),
                 "Discounts":("Discounts","sum"),"Net Sales":("Net Sales","sum"),
                 "Tax":("Tax","sum"),"Gross Profit":("Gross Profit","sum"),
                 "Transactions":("Transaction ID","nunique")}))
for c in ["Gross Sales","Discounts","Net Sales","Tax","Gross Profit"]: agg[c]=agg[c].round(2)
agg.to_csv("/home/claude/v2/square_item_sales_summary_by_store_24mo.csv", index=False)

ss = (df.groupby("Location", as_index=False)
        .agg(**{"Gross Sales":("Gross Sales","sum"),"Discounts":("Discounts","sum"),
                "Net Sales":("Net Sales","sum"),"Tax":("Tax","sum"),
                "Total Collected":("Total Collected","sum"),
                "Transactions":("Transaction ID","nunique"),"Items Sold":("Qty","sum")}))
for c in ["Gross Sales","Discounts","Net Sales","Tax","Total Collected"]: ss[c]=ss[c].round(2)
ss.to_csv("/home/claude/v2/square_sales_summary_by_store_24mo.csv", index=False)

# realized-parameter validation report (documentation must match reality)
pay_rows = df[df["Event Type"]=="Payment"]
daily = pay_rows.groupby("Date")["Net Sales"].sum()
rep = {
 "rows_total": len(df), "payments": int((df['Event Type']=='Payment').sum()),
 "refunds": int((df['Event Type']=='Refund').sum()),
 "date_min": df['Date'].min(), "date_max": df['Date'].max(),
 "distinct_days_with_sales": df['Date'].nunique(),
 "realized_discount_line_rate": round(float((pay_rows['Discounts']<0).mean()),4),
 "realized_card_share": round(float((pay_rows.drop_duplicates('Transaction ID')['Payment Method']=='Card').mean()),4),
 "realized_customer_id_rate": round(float((pay_rows.drop_duplicates('Transaction ID')['Customer ID']!='').mean()),4),
 "gross_total_$M": round(df['Gross Sales'].sum()/1e6,3),
 "top5_revenue_days": daily.sort_values(ascending=False).head(5).round(0).to_dict(),
 "christmas_rows": int((df['Date'].isin(['2024-12-25','2025-12-25'])).sum()),
 "yoy_growth_net": round(float(df[df['Date']>='2025-01-01']['Net Sales'].sum() /
                              df[df['Date']<'2025-01-01']['Net Sales'].sum()),4),
}
import json
with open("/home/claude/v2/validation_report.json","w") as f: json.dump(rep,f,indent=2,default=str)
print(json.dumps(rep,indent=2,default=str))
