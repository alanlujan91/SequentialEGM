/*!
 * reveal.js 4.3.1
 * https://revealjs.com
 * MIT licensed
 *
 * Copyright (C) 2011-2022 Hakim El Hattab, https://hakim.se
 */
!(function (e, t) {
  "object" == typeof exports && "undefined" != typeof module
    ? (module.exports = t())
    : "function" == typeof define && define.amd
    ? define(t)
    : ((e = "undefined" != typeof globalThis ? globalThis : e || self).Reveal =
        t());
})(this, function () {
  "use strict";
  var e =
      "undefined" != typeof globalThis
        ? globalThis
        : "undefined" != typeof window
        ? window
        : "undefined" != typeof global
        ? global
        : "undefined" != typeof self
        ? self
        : {},
    t = function (e) {
      return e && e.Math == Math && e;
    },
    n =
      t("object" == typeof globalThis && globalThis) ||
      t("object" == typeof window && window) ||
      t("object" == typeof self && self) ||
      t("object" == typeof e && e) ||
      (function () {
        return this;
      })() ||
      Function("return this")(),
    i = {},
    r = function (e) {
      try {
        return !!e();
      } catch (e) {
        return !0;
      }
    },
    a = !r(function () {
      return (
        7 !=
        Object.defineProperty({}, 1, {
          get: function () {
            return 7;
          },
        })[1]
      );
    }),
    o = {},
    s = {}.propertyIsEnumerable,
    l = Object.getOwnPropertyDescriptor,
    c = l && !s.call({ 1: 2 }, 1);
  o.f = c
    ? function (e) {
        var t = l(this, e);
        return !!t && t.enumerable;
      }
    : s;
  var u = function (e, t) {
      return {
        enumerable: !(1 & e),
        configurable: !(2 & e),
        writable: !(4 & e),
        value: t,
      };
    },
    d = {}.toString,
    h = function (e) {
      return d.call(e).slice(8, -1);
    },
    f = h,
    v = "".split,
    p = r(function () {
      return !Object("z").propertyIsEnumerable(0);
    })
      ? function (e) {
          return "String" == f(e) ? v.call(e, "") : Object(e);
        }
      : Object,
    g = function (e) {
      if (null == e) throw TypeError("Can't call method on " + e);
      return e;
    },
    m = p,
    y = g,
    b = function (e) {
      return m(y(e));
    },
    w = function (e) {
      return "object" == typeof e ? null !== e : "function" == typeof e;
    },
    E = w,
    S = function (e, t) {
      if (!E(e)) return e;
      var n, i;
      if (t && "function" == typeof (n = e.toString) && !E((i = n.call(e))))
        return i;
      if ("function" == typeof (n = e.valueOf) && !E((i = n.call(e)))) return i;
      if (!t && "function" == typeof (n = e.toString) && !E((i = n.call(e))))
        return i;
      throw TypeError("Can't convert object to primitive value");
    },
    k = g,
    A = function (e) {
      return Object(k(e));
    },
    R = A,
    x = {}.hasOwnProperty,
    L = function (e, t) {
      return x.call(R(e), t);
    },
    P = w,
    C = n.document,
    N = P(C) && P(C.createElement),
    M = function (e) {
      return N ? C.createElement(e) : {};
    },
    I = M,
    T =
      !a &&
      !r(function () {
        return (
          7 !=
          Object.defineProperty(I("div"), "a", {
            get: function () {
              return 7;
            },
          }).a
        );
      }),
    O = a,
    D = o,
    j = u,
    F = b,
    z = S,
    H = L,
    U = T,
    _ = Object.getOwnPropertyDescriptor;
  i.f = O
    ? _
    : function (e, t) {
        if (((e = F(e)), (t = z(t, !0)), U))
          try {
            return _(e, t);
          } catch (e) {}
        if (H(e, t)) return j(!D.f.call(e, t), e[t]);
      };
  var B = {},
    q = w,
    W = function (e) {
      if (!q(e)) throw TypeError(String(e) + " is not an object");
      return e;
    },
    V = a,
    K = T,
    Y = W,
    X = S,
    $ = Object.defineProperty;
  B.f = V
    ? $
    : function (e, t, n) {
        if ((Y(e), (t = X(t, !0)), Y(n), K))
          try {
            return $(e, t, n);
          } catch (e) {}
        if ("get" in n || "set" in n)
          throw TypeError("Accessors not supported");
        return "value" in n && (e[t] = n.value), e;
      };
  var G = B,
    J = u,
    Q = a
      ? function (e, t, n) {
          return G.f(e, t, J(1, n));
        }
      : function (e, t, n) {
          return (e[t] = n), e;
        },
    Z = { exports: {} },
    ee = n,
    te = Q,
    ne = function (e, t) {
      try {
        te(ee, e, t);
      } catch (n) {
        ee[e] = t;
      }
      return t;
    },
    ie = ne,
    re = "__core-js_shared__",
    ae = n[re] || ie(re, {}),
    oe = ae,
    se = Function.toString;
  "function" != typeof oe.inspectSource &&
    (oe.inspectSource = function (e) {
      return se.call(e);
    });
  var le = oe.inspectSource,
    ce = le,
    ue = n.WeakMap,
    de = "function" == typeof ue && /native code/.test(ce(ue)),
    he = { exports: {} },
    fe = ae;
  (he.exports = function (e, t) {
    return fe[e] || (fe[e] = void 0 !== t ? t : {});
  })("versions", []).push({
    version: "3.12.1",
    mode: "global",
    copyright: "© 2021 Denis Pushkarev (zloirock.ru)",
  });
  var ve,
    pe,
    ge,
    me = 0,
    ye = Math.random(),
    be = function (e) {
      return (
        "Symbol(" +
        String(void 0 === e ? "" : e) +
        ")_" +
        (++me + ye).toString(36)
      );
    },
    we = he.exports,
    Ee = be,
    Se = we("keys"),
    ke = function (e) {
      return Se[e] || (Se[e] = Ee(e));
    },
    Ae = {},
    Re = de,
    xe = w,
    Le = Q,
    Pe = L,
    Ce = ae,
    Ne = ke,
    Me = Ae,
    Ie = "Object already initialized",
    Te = n.WeakMap;
  if (Re || Ce.state) {
    var Oe = Ce.state || (Ce.state = new Te()),
      De = Oe.get,
      je = Oe.has,
      Fe = Oe.set;
    (ve = function (e, t) {
      if (je.call(Oe, e)) throw new TypeError(Ie);
      return (t.facade = e), Fe.call(Oe, e, t), t;
    }),
      (pe = function (e) {
        return De.call(Oe, e) || {};
      }),
      (ge = function (e) {
        return je.call(Oe, e);
      });
  } else {
    var ze = Ne("state");
    (Me[ze] = !0),
      (ve = function (e, t) {
        if (Pe(e, ze)) throw new TypeError(Ie);
        return (t.facade = e), Le(e, ze, t), t;
      }),
      (pe = function (e) {
        return Pe(e, ze) ? e[ze] : {};
      }),
      (ge = function (e) {
        return Pe(e, ze);
      });
  }
  var He = {
      set: ve,
      get: pe,
      has: ge,
      enforce: function (e) {
        return ge(e) ? pe(e) : ve(e, {});
      },
      getterFor: function (e) {
        return function (t) {
          var n;
          if (!xe(t) || (n = pe(t)).type !== e)
            throw TypeError("Incompatible receiver, " + e + " required");
          return n;
        };
      },
    },
    Ue = n,
    _e = Q,
    Be = L,
    qe = ne,
    We = le,
    Ve = He.get,
    Ke = He.enforce,
    Ye = String(String).split("String");
  (Z.exports = function (e, t, n, i) {
    var r,
      a = !!i && !!i.unsafe,
      o = !!i && !!i.enumerable,
      s = !!i && !!i.noTargetGet;
    "function" == typeof n &&
      ("string" != typeof t || Be(n, "name") || _e(n, "name", t),
      (r = Ke(n)).source ||
        (r.source = Ye.join("string" == typeof t ? t : ""))),
      e !== Ue
        ? (a ? !s && e[t] && (o = !0) : delete e[t],
          o ? (e[t] = n) : _e(e, t, n))
        : o
        ? (e[t] = n)
        : qe(t, n);
  })(Function.prototype, "toString", function () {
    return ("function" == typeof this && Ve(this).source) || We(this);
  });
  var Xe = n,
    $e = Xe,
    Ge = n,
    Je = function (e) {
      return "function" == typeof e ? e : void 0;
    },
    Qe = function (e, t) {
      return arguments.length < 2
        ? Je($e[e]) || Je(Ge[e])
        : ($e[e] && $e[e][t]) || (Ge[e] && Ge[e][t]);
    },
    Ze = {},
    et = Math.ceil,
    tt = Math.floor,
    nt = function (e) {
      return isNaN((e = +e)) ? 0 : (e > 0 ? tt : et)(e);
    },
    it = nt,
    rt = Math.min,
    at = function (e) {
      return e > 0 ? rt(it(e), 9007199254740991) : 0;
    },
    ot = nt,
    st = Math.max,
    lt = Math.min,
    ct = function (e, t) {
      var n = ot(e);
      return n < 0 ? st(n + t, 0) : lt(n, t);
    },
    ut = b,
    dt = at,
    ht = ct,
    ft = function (e) {
      return function (t, n, i) {
        var r,
          a = ut(t),
          o = dt(a.length),
          s = ht(i, o);
        if (e && n != n) {
          for (; o > s; ) if ((r = a[s++]) != r) return !0;
        } else
          for (; o > s; s++)
            if ((e || s in a) && a[s] === n) return e || s || 0;
        return !e && -1;
      };
    },
    vt = { includes: ft(!0), indexOf: ft(!1) },
    pt = L,
    gt = b,
    mt = vt.indexOf,
    yt = Ae,
    bt = function (e, t) {
      var n,
        i = gt(e),
        r = 0,
        a = [];
      for (n in i) !pt(yt, n) && pt(i, n) && a.push(n);
      for (; t.length > r; ) pt(i, (n = t[r++])) && (~mt(a, n) || a.push(n));
      return a;
    },
    wt = [
      "constructor",
      "hasOwnProperty",
      "isPrototypeOf",
      "propertyIsEnumerable",
      "toLocaleString",
      "toString",
      "valueOf",
    ],
    Et = bt,
    St = wt.concat("length", "prototype");
  Ze.f =
    Object.getOwnPropertyNames ||
    function (e) {
      return Et(e, St);
    };
  var kt = {};
  kt.f = Object.getOwnPropertySymbols;
  var At = Ze,
    Rt = kt,
    xt = W,
    Lt =
      Qe("Reflect", "ownKeys") ||
      function (e) {
        var t = At.f(xt(e)),
          n = Rt.f;
        return n ? t.concat(n(e)) : t;
      },
    Pt = L,
    Ct = Lt,
    Nt = i,
    Mt = B,
    It = function (e, t) {
      for (var n = Ct(t), i = Mt.f, r = Nt.f, a = 0; a < n.length; a++) {
        var o = n[a];
        Pt(e, o) || i(e, o, r(t, o));
      }
    },
    Tt = r,
    Ot = /#|\.prototype\./,
    Dt = function (e, t) {
      var n = Ft[jt(e)];
      return n == Ht || (n != zt && ("function" == typeof t ? Tt(t) : !!t));
    },
    jt = (Dt.normalize = function (e) {
      return String(e).replace(Ot, ".").toLowerCase();
    }),
    Ft = (Dt.data = {}),
    zt = (Dt.NATIVE = "N"),
    Ht = (Dt.POLYFILL = "P"),
    Ut = Dt,
    _t = n,
    Bt = i.f,
    qt = Q,
    Wt = Z.exports,
    Vt = ne,
    Kt = It,
    Yt = Ut,
    Xt = function (e, t) {
      var n,
        i,
        r,
        a,
        o,
        s = e.target,
        l = e.global,
        c = e.stat;
      if ((n = l ? _t : c ? _t[s] || Vt(s, {}) : (_t[s] || {}).prototype))
        for (i in t) {
          if (
            ((a = t[i]),
            (r = e.noTargetGet ? (o = Bt(n, i)) && o.value : n[i]),
            !Yt(l ? i : s + (c ? "." : "#") + i, e.forced) && void 0 !== r)
          ) {
            if (typeof a == typeof r) continue;
            Kt(a, r);
          }
          (e.sham || (r && r.sham)) && qt(a, "sham", !0), Wt(n, i, a, e);
        }
    },
    $t = bt,
    Gt = wt,
    Jt =
      Object.keys ||
      function (e) {
        return $t(e, Gt);
      },
    Qt = a,
    Zt = r,
    en = Jt,
    tn = kt,
    nn = o,
    rn = A,
    an = p,
    on = Object.assign,
    sn = Object.defineProperty,
    ln =
      !on ||
      Zt(function () {
        if (
          Qt &&
          1 !==
            on(
              { b: 1 },
              on(
                sn({}, "a", {
                  enumerable: !0,
                  get: function () {
                    sn(this, "b", { value: 3, enumerable: !1 });
                  },
                }),
                { b: 2 },
              ),
            ).b
        )
          return !0;
        var e = {},
          t = {},
          n = Symbol(),
          i = "abcdefghijklmnopqrst";
        return (
          (e[n] = 7),
          i.split("").forEach(function (e) {
            t[e] = e;
          }),
          7 != on({}, e)[n] || en(on({}, t)).join("") != i
        );
      })
        ? function (e, t) {
            for (
              var n = rn(e), i = arguments.length, r = 1, a = tn.f, o = nn.f;
              i > r;

            )
              for (
                var s,
                  l = an(arguments[r++]),
                  c = a ? en(l).concat(a(l)) : en(l),
                  u = c.length,
                  d = 0;
                u > d;

              )
                (s = c[d++]), (Qt && !o.call(l, s)) || (n[s] = l[s]);
            return n;
          }
        : on;
  Xt(
    { target: "Object", stat: !0, forced: Object.assign !== ln },
    { assign: ln },
  );
  var cn,
    un,
    dn = function (e) {
      if ("function" != typeof e)
        throw TypeError(String(e) + " is not a function");
      return e;
    },
    hn = dn,
    fn = function (e, t, n) {
      if ((hn(e), void 0 === t)) return e;
      switch (n) {
        case 0:
          return function () {
            return e.call(t);
          };
        case 1:
          return function (n) {
            return e.call(t, n);
          };
        case 2:
          return function (n, i) {
            return e.call(t, n, i);
          };
        case 3:
          return function (n, i, r) {
            return e.call(t, n, i, r);
          };
      }
      return function () {
        return e.apply(t, arguments);
      };
    },
    vn = h,
    pn =
      Array.isArray ||
      function (e) {
        return "Array" == vn(e);
      },
    gn = Qe("navigator", "userAgent") || "",
    mn = gn,
    yn = n.process,
    bn = yn && yn.versions,
    wn = bn && bn.v8;
  wn
    ? (un = (cn = wn.split("."))[0] < 4 ? 1 : cn[0] + cn[1])
    : mn &&
      (!(cn = mn.match(/Edge\/(\d+)/)) || cn[1] >= 74) &&
      (cn = mn.match(/Chrome\/(\d+)/)) &&
      (un = cn[1]);
  var En = un && +un,
    Sn = En,
    kn = r,
    An =
      !!Object.getOwnPropertySymbols &&
      !kn(function () {
        return !String(Symbol()) || (!Symbol.sham && Sn && Sn < 41);
      }),
    Rn = An && !Symbol.sham && "symbol" == typeof Symbol.iterator,
    xn = n,
    Ln = he.exports,
    Pn = L,
    Cn = be,
    Nn = An,
    Mn = Rn,
    In = Ln("wks"),
    Tn = xn.Symbol,
    On = Mn ? Tn : (Tn && Tn.withoutSetter) || Cn,
    Dn = function (e) {
      return (
        (Pn(In, e) && (Nn || "string" == typeof In[e])) ||
          (Nn && Pn(Tn, e) ? (In[e] = Tn[e]) : (In[e] = On("Symbol." + e))),
        In[e]
      );
    },
    jn = w,
    Fn = pn,
    zn = Dn("species"),
    Hn = function (e, t) {
      var n;
      return (
        Fn(e) &&
          ("function" != typeof (n = e.constructor) ||
          (n !== Array && !Fn(n.prototype))
            ? jn(n) && null === (n = n[zn]) && (n = void 0)
            : (n = void 0)),
        new (void 0 === n ? Array : n)(0 === t ? 0 : t)
      );
    },
    Un = fn,
    _n = p,
    Bn = A,
    qn = at,
    Wn = Hn,
    Vn = [].push,
    Kn = function (e) {
      var t = 1 == e,
        n = 2 == e,
        i = 3 == e,
        r = 4 == e,
        a = 6 == e,
        o = 7 == e,
        s = 5 == e || a;
      return function (l, c, u, d) {
        for (
          var h,
            f,
            v = Bn(l),
            p = _n(v),
            g = Un(c, u, 3),
            m = qn(p.length),
            y = 0,
            b = d || Wn,
            w = t ? b(l, m) : n || o ? b(l, 0) : void 0;
          m > y;
          y++
        )
          if ((s || y in p) && ((f = g((h = p[y]), y, v)), e))
            if (t) w[y] = f;
            else if (f)
              switch (e) {
                case 3:
                  return !0;
                case 5:
                  return h;
                case 6:
                  return y;
                case 2:
                  Vn.call(w, h);
              }
            else
              switch (e) {
                case 4:
                  return !1;
                case 7:
                  Vn.call(w, h);
              }
        return a ? -1 : i || r ? r : w;
      };
    },
    Yn = {
      forEach: Kn(0),
      map: Kn(1),
      filter: Kn(2),
      some: Kn(3),
      every: Kn(4),
      find: Kn(5),
      findIndex: Kn(6),
      filterOut: Kn(7),
    },
    Xn = r,
    $n = En,
    Gn = Dn("species"),
    Jn = function (e) {
      return (
        $n >= 51 ||
        !Xn(function () {
          var t = [];
          return (
            ((t.constructor = {})[Gn] = function () {
              return { foo: 1 };
            }),
            1 !== t[e](Boolean).foo
          );
        })
      );
    },
    Qn = Yn.map;
  Xt(
    { target: "Array", proto: !0, forced: !Jn("map") },
    {
      map: function (e) {
        return Qn(this, e, arguments.length > 1 ? arguments[1] : void 0);
      },
    },
  );
  var Zn = S,
    ei = B,
    ti = u,
    ni = function (e, t, n) {
      var i = Zn(t);
      i in e ? ei.f(e, i, ti(0, n)) : (e[i] = n);
    },
    ii = Xt,
    ri = r,
    ai = pn,
    oi = w,
    si = A,
    li = at,
    ci = ni,
    ui = Hn,
    di = Jn,
    hi = En,
    fi = Dn("isConcatSpreadable"),
    vi = 9007199254740991,
    pi = "Maximum allowed index exceeded",
    gi =
      hi >= 51 ||
      !ri(function () {
        var e = [];
        return (e[fi] = !1), e.concat()[0] !== e;
      }),
    mi = di("concat"),
    yi = function (e) {
      if (!oi(e)) return !1;
      var t = e[fi];
      return void 0 !== t ? !!t : ai(e);
    };
  function bi(e, t) {
    var n = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var i = Object.getOwnPropertySymbols(e);
      t &&
        (i = i.filter(function (t) {
          return Object.getOwnPropertyDescriptor(e, t).enumerable;
        })),
        n.push.apply(n, i);
    }
    return n;
  }
  function wi(e) {
    for (var t = 1; t < arguments.length; t++) {
      var n = null != arguments[t] ? arguments[t] : {};
      t % 2
        ? bi(Object(n), !0).forEach(function (t) {
            xi(e, t, n[t]);
          })
        : Object.getOwnPropertyDescriptors
        ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(n))
        : bi(Object(n)).forEach(function (t) {
            Object.defineProperty(e, t, Object.getOwnPropertyDescriptor(n, t));
          });
    }
    return e;
  }
  function Ei(e) {
    return (Ei =
      "function" == typeof Symbol && "symbol" == typeof Symbol.iterator
        ? function (e) {
            return typeof e;
          }
        : function (e) {
            return e &&
              "function" == typeof Symbol &&
              e.constructor === Symbol &&
              e !== Symbol.prototype
              ? "symbol"
              : typeof e;
          })(e);
  }
  function Si(e, t, n, i, r, a, o) {
    try {
      var s = e[a](o),
        l = s.value;
    } catch (e) {
      return void n(e);
    }
    s.done ? t(l) : Promise.resolve(l).then(i, r);
  }
  function ki(e, t) {
    if (!(e instanceof t))
      throw new TypeError("Cannot call a class as a function");
  }
  function Ai(e, t) {
    for (var n = 0; n < t.length; n++) {
      var i = t[n];
      (i.enumerable = i.enumerable || !1),
        (i.configurable = !0),
        "value" in i && (i.writable = !0),
        Object.defineProperty(e, i.key, i);
    }
  }
  function Ri(e, t, n) {
    return t && Ai(e.prototype, t), n && Ai(e, n), e;
  }
  function xi(e, t, n) {
    return (
      t in e
        ? Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          })
        : (e[t] = n),
      e
    );
  }
  function Li(e) {
    return (
      (function (e) {
        if (Array.isArray(e)) return Pi(e);
      })(e) ||
      (function (e) {
        if (
          ("undefined" != typeof Symbol && null != e[Symbol.iterator]) ||
          null != e["@@iterator"]
        )
          return Array.from(e);
      })(e) ||
      (function (e, t) {
        if (!e) return;
        if ("string" == typeof e) return Pi(e, t);
        var n = Object.prototype.toString.call(e).slice(8, -1);
        "Object" === n && e.constructor && (n = e.constructor.name);
        if ("Map" === n || "Set" === n) return Array.from(e);
        if (
          "Arguments" === n ||
          /^(?:Ui|I)nt(?:8|16|32)(?:Clamped)?Array$/.test(n)
        )
          return Pi(e, t);
      })(e) ||
      (function () {
        throw new TypeError(
          "Invalid attempt to spread non-iterable instance.\nIn order to be iterable, non-array objects must have a [Symbol.iterator]() method.",
        );
      })()
    );
  }
  function Pi(e, t) {
    (null == t || t > e.length) && (t = e.length);
    for (var n = 0, i = new Array(t); n < t; n++) i[n] = e[n];
    return i;
  }
  ii(
    { target: "Array", proto: !0, forced: !gi || !mi },
    {
      concat: function (e) {
        var t,
          n,
          i,
          r,
          a,
          o = si(this),
          s = ui(o, 0),
          l = 0;
        for (t = -1, i = arguments.length; t < i; t++)
          if (yi((a = -1 === t ? o : arguments[t]))) {
            if (l + (r = li(a.length)) > vi) throw TypeError(pi);
            for (n = 0; n < r; n++, l++) n in a && ci(s, l, a[n]);
          } else {
            if (l >= vi) throw TypeError(pi);
            ci(s, l++, a);
          }
        return (s.length = l), s;
      },
    },
  );
  var Ci = {};
  Ci[Dn("toStringTag")] = "z";
  var Ni = "[object z]" === String(Ci),
    Mi = Ni,
    Ii = h,
    Ti = Dn("toStringTag"),
    Oi =
      "Arguments" ==
      Ii(
        (function () {
          return arguments;
        })(),
      ),
    Di = Mi
      ? Ii
      : function (e) {
          var t, n, i;
          return void 0 === e
            ? "Undefined"
            : null === e
            ? "Null"
            : "string" ==
              typeof (n = (function (e, t) {
                try {
                  return e[t];
                } catch (e) {}
              })((t = Object(e)), Ti))
            ? n
            : Oi
            ? Ii(t)
            : "Object" == (i = Ii(t)) && "function" == typeof t.callee
            ? "Arguments"
            : i;
        },
    ji = Di,
    Fi = Ni
      ? {}.toString
      : function () {
          return "[object " + ji(this) + "]";
        },
    zi = Ni,
    Hi = Z.exports,
    Ui = Fi;
  zi || Hi(Object.prototype, "toString", Ui, { unsafe: !0 });
  var _i = n.Promise,
    Bi = Z.exports,
    qi = w,
    Wi = W,
    Vi = function (e) {
      if (!qi(e) && null !== e)
        throw TypeError("Can't set " + String(e) + " as a prototype");
      return e;
    },
    Ki =
      Object.setPrototypeOf ||
      ("__proto__" in {}
        ? (function () {
            var e,
              t = !1,
              n = {};
            try {
              (e = Object.getOwnPropertyDescriptor(
                Object.prototype,
                "__proto__",
              ).set).call(n, []),
                (t = n instanceof Array);
            } catch (e) {}
            return function (n, i) {
              return Wi(n), Vi(i), t ? e.call(n, i) : (n.__proto__ = i), n;
            };
          })()
        : void 0),
    Yi = B.f,
    Xi = L,
    $i = Dn("toStringTag"),
    Gi = function (e, t, n) {
      e &&
        !Xi((e = n ? e : e.prototype), $i) &&
        Yi(e, $i, { configurable: !0, value: t });
    },
    Ji = Qe,
    Qi = B,
    Zi = a,
    er = Dn("species"),
    tr = {},
    nr = tr,
    ir = Dn("iterator"),
    rr = Array.prototype,
    ar = function (e) {
      return void 0 !== e && (nr.Array === e || rr[ir] === e);
    },
    or = Di,
    sr = tr,
    lr = Dn("iterator"),
    cr = function (e) {
      if (null != e) return e[lr] || e["@@iterator"] || sr[or(e)];
    },
    ur = W,
    dr = function (e) {
      var t = e.return;
      if (void 0 !== t) return ur(t.call(e)).value;
    },
    hr = W,
    fr = ar,
    vr = at,
    pr = fn,
    gr = cr,
    mr = dr,
    yr = function (e, t) {
      (this.stopped = e), (this.result = t);
    },
    br = Dn("iterator"),
    wr = !1;
  try {
    var Er = 0,
      Sr = {
        next: function () {
          return { done: !!Er++ };
        },
        return: function () {
          wr = !0;
        },
      };
    (Sr[br] = function () {
      return this;
    }),
      Array.from(Sr, function () {
        throw 2;
      });
  } catch (e) {}
  var kr,
    Ar,
    Rr,
    xr = function (e, t) {
      if (!t && !wr) return !1;
      var n = !1;
      try {
        var i = {};
        (i[br] = function () {
          return {
            next: function () {
              return { done: (n = !0) };
            },
          };
        }),
          e(i);
      } catch (e) {}
      return n;
    },
    Lr = W,
    Pr = dn,
    Cr = Dn("species"),
    Nr = function (e, t) {
      var n,
        i = Lr(e).constructor;
      return void 0 === i || null == (n = Lr(i)[Cr]) ? t : Pr(n);
    },
    Mr = Qe("document", "documentElement"),
    Ir = /(?:iphone|ipod|ipad).*applewebkit/i.test(gn),
    Tr = "process" == h(n.process),
    Or = n,
    Dr = r,
    jr = fn,
    Fr = Mr,
    zr = M,
    Hr = Ir,
    Ur = Tr,
    _r = Or.location,
    Br = Or.setImmediate,
    qr = Or.clearImmediate,
    Wr = Or.process,
    Vr = Or.MessageChannel,
    Kr = Or.Dispatch,
    Yr = 0,
    Xr = {},
    $r = "onreadystatechange",
    Gr = function (e) {
      if (Xr.hasOwnProperty(e)) {
        var t = Xr[e];
        delete Xr[e], t();
      }
    },
    Jr = function (e) {
      return function () {
        Gr(e);
      };
    },
    Qr = function (e) {
      Gr(e.data);
    },
    Zr = function (e) {
      Or.postMessage(e + "", _r.protocol + "//" + _r.host);
    };
  (Br && qr) ||
    ((Br = function (e) {
      for (var t = [], n = 1; arguments.length > n; ) t.push(arguments[n++]);
      return (
        (Xr[++Yr] = function () {
          ("function" == typeof e ? e : Function(e)).apply(void 0, t);
        }),
        kr(Yr),
        Yr
      );
    }),
    (qr = function (e) {
      delete Xr[e];
    }),
    Ur
      ? (kr = function (e) {
          Wr.nextTick(Jr(e));
        })
      : Kr && Kr.now
      ? (kr = function (e) {
          Kr.now(Jr(e));
        })
      : Vr && !Hr
      ? ((Rr = (Ar = new Vr()).port2),
        (Ar.port1.onmessage = Qr),
        (kr = jr(Rr.postMessage, Rr, 1)))
      : Or.addEventListener &&
        "function" == typeof postMessage &&
        !Or.importScripts &&
        _r &&
        "file:" !== _r.protocol &&
        !Dr(Zr)
      ? ((kr = Zr), Or.addEventListener("message", Qr, !1))
      : (kr =
          $r in zr("script")
            ? function (e) {
                Fr.appendChild(zr("script")).onreadystatechange = function () {
                  Fr.removeChild(this), Gr(e);
                };
              }
            : function (e) {
                setTimeout(Jr(e), 0);
              }));
  var ea,
    ta,
    na,
    ia,
    ra,
    aa,
    oa,
    sa,
    la = { set: Br, clear: qr },
    ca = /web0s(?!.*chrome)/i.test(gn),
    ua = n,
    da = i.f,
    ha = la.set,
    fa = Ir,
    va = ca,
    pa = Tr,
    ga = ua.MutationObserver || ua.WebKitMutationObserver,
    ma = ua.document,
    ya = ua.process,
    ba = ua.Promise,
    wa = da(ua, "queueMicrotask"),
    Ea = wa && wa.value;
  Ea ||
    ((ea = function () {
      var e, t;
      for (pa && (e = ya.domain) && e.exit(); ta; ) {
        (t = ta.fn), (ta = ta.next);
        try {
          t();
        } catch (e) {
          throw (ta ? ia() : (na = void 0), e);
        }
      }
      (na = void 0), e && e.enter();
    }),
    fa || pa || va || !ga || !ma
      ? ba && ba.resolve
        ? (((oa = ba.resolve(void 0)).constructor = ba),
          (sa = oa.then),
          (ia = function () {
            sa.call(oa, ea);
          }))
        : (ia = pa
            ? function () {
                ya.nextTick(ea);
              }
            : function () {
                ha.call(ua, ea);
              })
      : ((ra = !0),
        (aa = ma.createTextNode("")),
        new ga(ea).observe(aa, { characterData: !0 }),
        (ia = function () {
          aa.data = ra = !ra;
        })));
  var Sa =
      Ea ||
      function (e) {
        var t = { fn: e, next: void 0 };
        na && (na.next = t), ta || ((ta = t), ia()), (na = t);
      },
    ka = {},
    Aa = dn,
    Ra = function (e) {
      var t, n;
      (this.promise = new e(function (e, i) {
        if (void 0 !== t || void 0 !== n)
          throw TypeError("Bad Promise constructor");
        (t = e), (n = i);
      })),
        (this.resolve = Aa(t)),
        (this.reject = Aa(n));
    };
  ka.f = function (e) {
    return new Ra(e);
  };
  var xa,
    La,
    Pa,
    Ca,
    Na = W,
    Ma = w,
    Ia = ka,
    Ta = n,
    Oa = "object" == typeof window,
    Da = Xt,
    ja = n,
    Fa = Qe,
    za = _i,
    Ha = Z.exports,
    Ua = function (e, t, n) {
      for (var i in t) Bi(e, i, t[i], n);
      return e;
    },
    _a = Ki,
    Ba = Gi,
    qa = function (e) {
      var t = Ji(e),
        n = Qi.f;
      Zi &&
        t &&
        !t[er] &&
        n(t, er, {
          configurable: !0,
          get: function () {
            return this;
          },
        });
    },
    Wa = w,
    Va = dn,
    Ka = function (e, t, n) {
      if (!(e instanceof t))
        throw TypeError("Incorrect " + (n ? n + " " : "") + "invocation");
      return e;
    },
    Ya = le,
    Xa = function (e, t, n) {
      var i,
        r,
        a,
        o,
        s,
        l,
        c,
        u = n && n.that,
        d = !(!n || !n.AS_ENTRIES),
        h = !(!n || !n.IS_ITERATOR),
        f = !(!n || !n.INTERRUPTED),
        v = pr(t, u, 1 + d + f),
        p = function (e) {
          return i && mr(i), new yr(!0, e);
        },
        g = function (e) {
          return d
            ? (hr(e), f ? v(e[0], e[1], p) : v(e[0], e[1]))
            : f
            ? v(e, p)
            : v(e);
        };
      if (h) i = e;
      else {
        if ("function" != typeof (r = gr(e)))
          throw TypeError("Target is not iterable");
        if (fr(r)) {
          for (a = 0, o = vr(e.length); o > a; a++)
            if ((s = g(e[a])) && s instanceof yr) return s;
          return new yr(!1);
        }
        i = r.call(e);
      }
      for (l = i.next; !(c = l.call(i)).done; ) {
        try {
          s = g(c.value);
        } catch (e) {
          throw (mr(i), e);
        }
        if ("object" == typeof s && s && s instanceof yr) return s;
      }
      return new yr(!1);
    },
    $a = xr,
    Ga = Nr,
    Ja = la.set,
    Qa = Sa,
    Za = function (e, t) {
      if ((Na(e), Ma(t) && t.constructor === e)) return t;
      var n = Ia.f(e);
      return (0, n.resolve)(t), n.promise;
    },
    eo = function (e, t) {
      var n = Ta.console;
      n && n.error && (1 === arguments.length ? n.error(e) : n.error(e, t));
    },
    to = ka,
    no = function (e) {
      try {
        return { error: !1, value: e() };
      } catch (e) {
        return { error: !0, value: e };
      }
    },
    io = He,
    ro = Ut,
    ao = Oa,
    oo = Tr,
    so = En,
    lo = Dn("species"),
    co = "Promise",
    uo = io.get,
    ho = io.set,
    fo = io.getterFor(co),
    vo = za && za.prototype,
    po = za,
    go = vo,
    mo = ja.TypeError,
    yo = ja.document,
    bo = ja.process,
    wo = to.f,
    Eo = wo,
    So = !!(yo && yo.createEvent && ja.dispatchEvent),
    ko = "function" == typeof PromiseRejectionEvent,
    Ao = "unhandledrejection",
    Ro = !1,
    xo = ro(co, function () {
      var e = Ya(po) !== String(po);
      if (!e && 66 === so) return !0;
      if (so >= 51 && /native code/.test(po)) return !1;
      var t = new po(function (e) {
          e(1);
        }),
        n = function (e) {
          e(
            function () {},
            function () {},
          );
        };
      return (
        ((t.constructor = {})[lo] = n),
        !(Ro = t.then(function () {}) instanceof n) || (!e && ao && !ko)
      );
    }),
    Lo =
      xo ||
      !$a(function (e) {
        po.all(e).catch(function () {});
      }),
    Po = function (e) {
      var t;
      return !(!Wa(e) || "function" != typeof (t = e.then)) && t;
    },
    Co = function (e, t) {
      if (!e.notified) {
        e.notified = !0;
        var n = e.reactions;
        Qa(function () {
          for (var i = e.value, r = 1 == e.state, a = 0; n.length > a; ) {
            var o,
              s,
              l,
              c = n[a++],
              u = r ? c.ok : c.fail,
              d = c.resolve,
              h = c.reject,
              f = c.domain;
            try {
              u
                ? (r || (2 === e.rejection && To(e), (e.rejection = 1)),
                  !0 === u
                    ? (o = i)
                    : (f && f.enter(), (o = u(i)), f && (f.exit(), (l = !0))),
                  o === c.promise
                    ? h(mo("Promise-chain cycle"))
                    : (s = Po(o))
                    ? s.call(o, d, h)
                    : d(o))
                : h(i);
            } catch (e) {
              f && !l && f.exit(), h(e);
            }
          }
          (e.reactions = []), (e.notified = !1), t && !e.rejection && Mo(e);
        });
      }
    },
    No = function (e, t, n) {
      var i, r;
      So
        ? (((i = yo.createEvent("Event")).promise = t),
          (i.reason = n),
          i.initEvent(e, !1, !0),
          ja.dispatchEvent(i))
        : (i = { promise: t, reason: n }),
        !ko && (r = ja["on" + e])
          ? r(i)
          : e === Ao && eo("Unhandled promise rejection", n);
    },
    Mo = function (e) {
      Ja.call(ja, function () {
        var t,
          n = e.facade,
          i = e.value;
        if (
          Io(e) &&
          ((t = no(function () {
            oo ? bo.emit("unhandledRejection", i, n) : No(Ao, n, i);
          })),
          (e.rejection = oo || Io(e) ? 2 : 1),
          t.error)
        )
          throw t.value;
      });
    },
    Io = function (e) {
      return 1 !== e.rejection && !e.parent;
    },
    To = function (e) {
      Ja.call(ja, function () {
        var t = e.facade;
        oo
          ? bo.emit("rejectionHandled", t)
          : No("rejectionhandled", t, e.value);
      });
    },
    Oo = function (e, t, n) {
      return function (i) {
        e(t, i, n);
      };
    },
    Do = function (e, t, n) {
      e.done ||
        ((e.done = !0), n && (e = n), (e.value = t), (e.state = 2), Co(e, !0));
    },
    jo = function (e, t, n) {
      if (!e.done) {
        (e.done = !0), n && (e = n);
        try {
          if (e.facade === t) throw mo("Promise can't be resolved itself");
          var i = Po(t);
          i
            ? Qa(function () {
                var n = { done: !1 };
                try {
                  i.call(t, Oo(jo, n, e), Oo(Do, n, e));
                } catch (t) {
                  Do(n, t, e);
                }
              })
            : ((e.value = t), (e.state = 1), Co(e, !1));
        } catch (t) {
          Do({ done: !1 }, t, e);
        }
      }
    };
  if (
    xo &&
    ((go = (po = function (e) {
      Ka(this, po, co), Va(e), xa.call(this);
      var t = uo(this);
      try {
        e(Oo(jo, t), Oo(Do, t));
      } catch (e) {
        Do(t, e);
      }
    }).prototype),
    ((xa = function (e) {
      ho(this, {
        type: co,
        done: !1,
        notified: !1,
        parent: !1,
        reactions: [],
        rejection: !1,
        state: 0,
        value: void 0,
      });
    }).prototype = Ua(go, {
      then: function (e, t) {
        var n = fo(this),
          i = wo(Ga(this, po));
        return (
          (i.ok = "function" != typeof e || e),
          (i.fail = "function" == typeof t && t),
          (i.domain = oo ? bo.domain : void 0),
          (n.parent = !0),
          n.reactions.push(i),
          0 != n.state && Co(n, !1),
          i.promise
        );
      },
      catch: function (e) {
        return this.then(void 0, e);
      },
    })),
    (La = function () {
      var e = new xa(),
        t = uo(e);
      (this.promise = e), (this.resolve = Oo(jo, t)), (this.reject = Oo(Do, t));
    }),
    (to.f = wo =
      function (e) {
        return e === po || e === Pa ? new La(e) : Eo(e);
      }),
    "function" == typeof za && vo !== Object.prototype)
  ) {
    (Ca = vo.then),
      Ro ||
        (Ha(
          vo,
          "then",
          function (e, t) {
            var n = this;
            return new po(function (e, t) {
              Ca.call(n, e, t);
            }).then(e, t);
          },
          { unsafe: !0 },
        ),
        Ha(vo, "catch", go.catch, { unsafe: !0 }));
    try {
      delete vo.constructor;
    } catch (e) {}
    _a && _a(vo, go);
  }
  Da({ global: !0, wrap: !0, forced: xo }, { Promise: po }),
    Ba(po, co, !1),
    qa(co),
    (Pa = Fa(co)),
    Da(
      { target: co, stat: !0, forced: xo },
      {
        reject: function (e) {
          var t = wo(this);
          return t.reject.call(void 0, e), t.promise;
        },
      },
    ),
    Da(
      { target: co, stat: !0, forced: xo },
      {
        resolve: function (e) {
          return Za(this, e);
        },
      },
    ),
    Da(
      { target: co, stat: !0, forced: Lo },
      {
        all: function (e) {
          var t = this,
            n = wo(t),
            i = n.resolve,
            r = n.reject,
            a = no(function () {
              var n = Va(t.resolve),
                a = [],
                o = 0,
                s = 1;
              Xa(e, function (e) {
                var l = o++,
                  c = !1;
                a.push(void 0),
                  s++,
                  n.call(t, e).then(function (e) {
                    c || ((c = !0), (a[l] = e), --s || i(a));
                  }, r);
              }),
                --s || i(a);
            });
          return a.error && r(a.value), n.promise;
        },
        race: function (e) {
          var t = this,
            n = wo(t),
            i = n.reject,
            r = no(function () {
              var r = Va(t.resolve);
              Xa(e, function (e) {
                r.call(t, e).then(n.resolve, i);
              });
            });
          return r.error && i(r.value), n.promise;
        },
      },
    );
  var Fo = r,
    zo = function (e, t) {
      var n = [][e];
      return (
        !!n &&
        Fo(function () {
          n.call(
            null,
            t ||
              function () {
                throw 1;
              },
            1,
          );
        })
      );
    },
    Ho = Yn.forEach,
    Uo = n,
    _o = {
      CSSRuleList: 0,
      CSSStyleDeclaration: 0,
      CSSValueList: 0,
      ClientRectList: 0,
      DOMRectList: 0,
      DOMStringList: 0,
      DOMTokenList: 1,
      DataTransferItemList: 0,
      FileList: 0,
      HTMLAllCollection: 0,
      HTMLCollection: 0,
      HTMLFormElement: 0,
      HTMLSelectElement: 0,
      MediaList: 0,
      MimeTypeArray: 0,
      NamedNodeMap: 0,
      NodeList: 1,
      PaintRequestList: 0,
      Plugin: 0,
      PluginArray: 0,
      SVGLengthList: 0,
      SVGNumberList: 0,
      SVGPathSegList: 0,
      SVGPointList: 0,
      SVGStringList: 0,
      SVGTransformList: 0,
      SourceBufferList: 0,
      StyleSheetList: 0,
      TextTrackCueList: 0,
      TextTrackList: 0,
      TouchList: 0,
    },
    Bo = zo("forEach")
      ? [].forEach
      : function (e) {
          return Ho(this, e, arguments.length > 1 ? arguments[1] : void 0);
        },
    qo = Q;
  for (var Wo in _o) {
    var Vo = Uo[Wo],
      Ko = Vo && Vo.prototype;
    if (Ko && Ko.forEach !== Bo)
      try {
        qo(Ko, "forEach", Bo);
      } catch (e) {
        Ko.forEach = Bo;
      }
  }
  var Yo = W,
    Xo = dr,
    $o = fn,
    Go = A,
    Jo = function (e, t, n, i) {
      try {
        return i ? t(Yo(n)[0], n[1]) : t(n);
      } catch (t) {
        throw (Xo(e), t);
      }
    },
    Qo = ar,
    Zo = at,
    es = ni,
    ts = cr,
    ns = function (e) {
      var t,
        n,
        i,
        r,
        a,
        o,
        s = Go(e),
        l = "function" == typeof this ? this : Array,
        c = arguments.length,
        u = c > 1 ? arguments[1] : void 0,
        d = void 0 !== u,
        h = ts(s),
        f = 0;
      if (
        (d && (u = $o(u, c > 2 ? arguments[2] : void 0, 2)),
        null == h || (l == Array && Qo(h)))
      )
        for (n = new l((t = Zo(s.length))); t > f; f++)
          (o = d ? u(s[f], f) : s[f]), es(n, f, o);
      else
        for (a = (r = h.call(s)).next, n = new l(); !(i = a.call(r)).done; f++)
          (o = d ? Jo(r, u, [i.value, f], !0) : i.value), es(n, f, o);
      return (n.length = f), n;
    };
  Xt(
    {
      target: "Array",
      stat: !0,
      forced: !xr(function (e) {
        Array.from(e);
      }),
    },
    { from: ns },
  );
  var is,
    rs,
    as,
    os = nt,
    ss = g,
    ls = function (e) {
      return function (t, n) {
        var i,
          r,
          a = String(ss(t)),
          o = os(n),
          s = a.length;
        return o < 0 || o >= s
          ? e
            ? ""
            : void 0
          : (i = a.charCodeAt(o)) < 55296 ||
            i > 56319 ||
            o + 1 === s ||
            (r = a.charCodeAt(o + 1)) < 56320 ||
            r > 57343
          ? e
            ? a.charAt(o)
            : i
          : e
          ? a.slice(o, o + 2)
          : r - 56320 + ((i - 55296) << 10) + 65536;
      };
    },
    cs = { codeAt: ls(!1), charAt: ls(!0) },
    us = !r(function () {
      function e() {}
      return (
        (e.prototype.constructor = null),
        Object.getPrototypeOf(new e()) !== e.prototype
      );
    }),
    ds = L,
    hs = A,
    fs = us,
    vs = ke("IE_PROTO"),
    ps = Object.prototype,
    gs = fs
      ? Object.getPrototypeOf
      : function (e) {
          return (
            (e = hs(e)),
            ds(e, vs)
              ? e[vs]
              : "function" == typeof e.constructor && e instanceof e.constructor
              ? e.constructor.prototype
              : e instanceof Object
              ? ps
              : null
          );
        },
    ms = r,
    ys = gs,
    bs = Q,
    ws = L,
    Es = Dn("iterator"),
    Ss = !1;
  [].keys &&
    ("next" in (as = [].keys())
      ? (rs = ys(ys(as))) !== Object.prototype && (is = rs)
      : (Ss = !0)),
    (null == is ||
      ms(function () {
        var e = {};
        return is[Es].call(e) !== e;
      })) &&
      (is = {}),
    ws(is, Es) ||
      bs(is, Es, function () {
        return this;
      });
  var ks,
    As = { IteratorPrototype: is, BUGGY_SAFARI_ITERATORS: Ss },
    Rs = B,
    xs = W,
    Ls = Jt,
    Ps = a
      ? Object.defineProperties
      : function (e, t) {
          xs(e);
          for (var n, i = Ls(t), r = i.length, a = 0; r > a; )
            Rs.f(e, (n = i[a++]), t[n]);
          return e;
        },
    Cs = W,
    Ns = Ps,
    Ms = wt,
    Is = Ae,
    Ts = Mr,
    Os = M,
    Ds = ke("IE_PROTO"),
    js = function () {},
    Fs = function (e) {
      return "<script>" + e + "</" + "script>";
    },
    zs = function () {
      try {
        ks = document.domain && new ActiveXObject("htmlfile");
      } catch (e) {}
      var e, t;
      zs = ks
        ? (function (e) {
            e.write(Fs("")), e.close();
            var t = e.parentWindow.Object;
            return (e = null), t;
          })(ks)
        : (((t = Os("iframe")).style.display = "none"),
          Ts.appendChild(t),
          (t.src = String("javascript:")),
          (e = t.contentWindow.document).open(),
          e.write(Fs("document.F=Object")),
          e.close(),
          e.F);
      for (var n = Ms.length; n--; ) delete zs.prototype[Ms[n]];
      return zs();
    };
  Is[Ds] = !0;
  var Hs =
      Object.create ||
      function (e, t) {
        var n;
        return (
          null !== e
            ? ((js.prototype = Cs(e)),
              (n = new js()),
              (js.prototype = null),
              (n[Ds] = e))
            : (n = zs()),
          void 0 === t ? n : Ns(n, t)
        );
      },
    Us = As.IteratorPrototype,
    _s = Hs,
    Bs = u,
    qs = Gi,
    Ws = tr,
    Vs = function () {
      return this;
    },
    Ks = Xt,
    Ys = function (e, t, n) {
      var i = t + " Iterator";
      return (
        (e.prototype = _s(Us, { next: Bs(1, n) })),
        qs(e, i, !1),
        (Ws[i] = Vs),
        e
      );
    },
    Xs = gs,
    $s = Ki,
    Gs = Gi,
    Js = Q,
    Qs = Z.exports,
    Zs = tr,
    el = As.IteratorPrototype,
    tl = As.BUGGY_SAFARI_ITERATORS,
    nl = Dn("iterator"),
    il = "keys",
    rl = "values",
    al = "entries",
    ol = function () {
      return this;
    },
    sl = cs.charAt,
    ll = He,
    cl = function (e, t, n, i, r, a, o) {
      Ys(n, t, i);
      var s,
        l,
        c,
        u = function (e) {
          if (e === r && p) return p;
          if (!tl && e in f) return f[e];
          switch (e) {
            case il:
            case rl:
            case al:
              return function () {
                return new n(this, e);
              };
          }
          return function () {
            return new n(this);
          };
        },
        d = t + " Iterator",
        h = !1,
        f = e.prototype,
        v = f[nl] || f["@@iterator"] || (r && f[r]),
        p = (!tl && v) || u(r),
        g = ("Array" == t && f.entries) || v;
      if (
        (g &&
          ((s = Xs(g.call(new e()))),
          el !== Object.prototype &&
            s.next &&
            (Xs(s) !== el &&
              ($s ? $s(s, el) : "function" != typeof s[nl] && Js(s, nl, ol)),
            Gs(s, d, !0))),
        r == rl &&
          v &&
          v.name !== rl &&
          ((h = !0),
          (p = function () {
            return v.call(this);
          })),
        f[nl] !== p && Js(f, nl, p),
        (Zs[t] = p),
        r)
      )
        if (((l = { values: u(rl), keys: a ? p : u(il), entries: u(al) }), o))
          for (c in l) (tl || h || !(c in f)) && Qs(f, c, l[c]);
        else Ks({ target: t, proto: !0, forced: tl || h }, l);
      return l;
    },
    ul = "String Iterator",
    dl = ll.set,
    hl = ll.getterFor(ul);
  cl(
    String,
    "String",
    function (e) {
      dl(this, { type: ul, string: String(e), index: 0 });
    },
    function () {
      var e,
        t = hl(this),
        n = t.string,
        i = t.index;
      return i >= n.length
        ? { value: void 0, done: !0 }
        : ((e = sl(n, i)), (t.index += e.length), { value: e, done: !1 });
    },
  );
  var fl = "\t\n\v\f\r                　\u2028\u2029\ufeff",
    vl = g,
    pl = "[\t\n\v\f\r                　\u2028\u2029\ufeff]",
    gl = RegExp("^" + pl + pl + "*"),
    ml = RegExp(pl + pl + "*$"),
    yl = function (e) {
      return function (t) {
        var n = String(vl(t));
        return (
          1 & e && (n = n.replace(gl, "")), 2 & e && (n = n.replace(ml, "")), n
        );
      };
    },
    bl = { start: yl(1), end: yl(2), trim: yl(3) },
    wl = r,
    El = fl,
    Sl = bl.trim;
  Xt(
    {
      target: "String",
      proto: !0,
      forced: (function (e) {
        return wl(function () {
          return !!El[e]() || "​᠎" != "​᠎"[e]() || El[e].name !== e;
        });
      })("trim"),
    },
    {
      trim: function () {
        return Sl(this);
      },
    },
  );
  var kl = {},
    Al = b,
    Rl = Ze.f,
    xl = {}.toString,
    Ll =
      "object" == typeof window && window && Object.getOwnPropertyNames
        ? Object.getOwnPropertyNames(window)
        : [];
  kl.f = function (e) {
    return Ll && "[object Window]" == xl.call(e)
      ? (function (e) {
          try {
            return Rl(e);
          } catch (e) {
            return Ll.slice();
          }
        })(e)
      : Rl(Al(e));
  };
  var Pl = {},
    Cl = Dn;
  Pl.f = Cl;
  var Nl = Xe,
    Ml = L,
    Il = Pl,
    Tl = B.f,
    Ol = Xt,
    Dl = n,
    jl = Qe,
    Fl = a,
    zl = An,
    Hl = Rn,
    Ul = r,
    _l = L,
    Bl = pn,
    ql = w,
    Wl = W,
    Vl = A,
    Kl = b,
    Yl = S,
    Xl = u,
    $l = Hs,
    Gl = Jt,
    Jl = Ze,
    Ql = kl,
    Zl = kt,
    ec = i,
    tc = B,
    nc = o,
    ic = Q,
    rc = Z.exports,
    ac = he.exports,
    oc = Ae,
    sc = be,
    lc = Dn,
    cc = Pl,
    uc = function (e) {
      var t = Nl.Symbol || (Nl.Symbol = {});
      Ml(t, e) || Tl(t, e, { value: Il.f(e) });
    },
    dc = Gi,
    hc = He,
    fc = Yn.forEach,
    vc = ke("hidden"),
    pc = "Symbol",
    gc = lc("toPrimitive"),
    mc = hc.set,
    yc = hc.getterFor(pc),
    bc = Object.prototype,
    wc = Dl.Symbol,
    Ec = jl("JSON", "stringify"),
    Sc = ec.f,
    kc = tc.f,
    Ac = Ql.f,
    Rc = nc.f,
    xc = ac("symbols"),
    Lc = ac("op-symbols"),
    Pc = ac("string-to-symbol-registry"),
    Cc = ac("symbol-to-string-registry"),
    Nc = ac("wks"),
    Mc = Dl.QObject,
    Ic = !Mc || !Mc.prototype || !Mc.prototype.findChild,
    Tc =
      Fl &&
      Ul(function () {
        return (
          7 !=
          $l(
            kc({}, "a", {
              get: function () {
                return kc(this, "a", { value: 7 }).a;
              },
            }),
          ).a
        );
      })
        ? function (e, t, n) {
            var i = Sc(bc, t);
            i && delete bc[t], kc(e, t, n), i && e !== bc && kc(bc, t, i);
          }
        : kc,
    Oc = function (e, t) {
      var n = (xc[e] = $l(wc.prototype));
      return (
        mc(n, { type: pc, tag: e, description: t }),
        Fl || (n.description = t),
        n
      );
    },
    Dc = Hl
      ? function (e) {
          return "symbol" == typeof e;
        }
      : function (e) {
          return Object(e) instanceof wc;
        },
    jc = function (e, t, n) {
      e === bc && jc(Lc, t, n), Wl(e);
      var i = Yl(t, !0);
      return (
        Wl(n),
        _l(xc, i)
          ? (n.enumerable
              ? (_l(e, vc) && e[vc][i] && (e[vc][i] = !1),
                (n = $l(n, { enumerable: Xl(0, !1) })))
              : (_l(e, vc) || kc(e, vc, Xl(1, {})), (e[vc][i] = !0)),
            Tc(e, i, n))
          : kc(e, i, n)
      );
    },
    Fc = function (e, t) {
      Wl(e);
      var n = Kl(t),
        i = Gl(n).concat(_c(n));
      return (
        fc(i, function (t) {
          (Fl && !zc.call(n, t)) || jc(e, t, n[t]);
        }),
        e
      );
    },
    zc = function (e) {
      var t = Yl(e, !0),
        n = Rc.call(this, t);
      return (
        !(this === bc && _l(xc, t) && !_l(Lc, t)) &&
        (!(n || !_l(this, t) || !_l(xc, t) || (_l(this, vc) && this[vc][t])) ||
          n)
      );
    },
    Hc = function (e, t) {
      var n = Kl(e),
        i = Yl(t, !0);
      if (n !== bc || !_l(xc, i) || _l(Lc, i)) {
        var r = Sc(n, i);
        return (
          !r || !_l(xc, i) || (_l(n, vc) && n[vc][i]) || (r.enumerable = !0), r
        );
      }
    },
    Uc = function (e) {
      var t = Ac(Kl(e)),
        n = [];
      return (
        fc(t, function (e) {
          _l(xc, e) || _l(oc, e) || n.push(e);
        }),
        n
      );
    },
    _c = function (e) {
      var t = e === bc,
        n = Ac(t ? Lc : Kl(e)),
        i = [];
      return (
        fc(n, function (e) {
          !_l(xc, e) || (t && !_l(bc, e)) || i.push(xc[e]);
        }),
        i
      );
    };
  (zl ||
    (rc(
      (wc = function () {
        if (this instanceof wc) throw TypeError("Symbol is not a constructor");
        var e =
            arguments.length && void 0 !== arguments[0]
              ? String(arguments[0])
              : void 0,
          t = sc(e),
          n = function (e) {
            this === bc && n.call(Lc, e),
              _l(this, vc) && _l(this[vc], t) && (this[vc][t] = !1),
              Tc(this, t, Xl(1, e));
          };
        return Fl && Ic && Tc(bc, t, { configurable: !0, set: n }), Oc(t, e);
      }).prototype,
      "toString",
      function () {
        return yc(this).tag;
      },
    ),
    rc(wc, "withoutSetter", function (e) {
      return Oc(sc(e), e);
    }),
    (nc.f = zc),
    (tc.f = jc),
    (ec.f = Hc),
    (Jl.f = Ql.f = Uc),
    (Zl.f = _c),
    (cc.f = function (e) {
      return Oc(lc(e), e);
    }),
    Fl &&
      (kc(wc.prototype, "description", {
        configurable: !0,
        get: function () {
          return yc(this).description;
        },
      }),
      rc(bc, "propertyIsEnumerable", zc, { unsafe: !0 }))),
  Ol({ global: !0, wrap: !0, forced: !zl, sham: !zl }, { Symbol: wc }),
  fc(Gl(Nc), function (e) {
    uc(e);
  }),
  Ol(
    { target: pc, stat: !0, forced: !zl },
    {
      for: function (e) {
        var t = String(e);
        if (_l(Pc, t)) return Pc[t];
        var n = wc(t);
        return (Pc[t] = n), (Cc[n] = t), n;
      },
      keyFor: function (e) {
        if (!Dc(e)) throw TypeError(e + " is not a symbol");
        if (_l(Cc, e)) return Cc[e];
      },
      useSetter: function () {
        Ic = !0;
      },
      useSimple: function () {
        Ic = !1;
      },
    },
  ),
  Ol(
    { target: "Object", stat: !0, forced: !zl, sham: !Fl },
    {
      create: function (e, t) {
        return void 0 === t ? $l(e) : Fc($l(e), t);
      },
      defineProperty: jc,
      defineProperties: Fc,
      getOwnPropertyDescriptor: Hc,
    },
  ),
  Ol(
    { target: "Object", stat: !0, forced: !zl },
    { getOwnPropertyNames: Uc, getOwnPropertySymbols: _c },
  ),
  Ol(
    {
      target: "Object",
      stat: !0,
      forced: Ul(function () {
        Zl.f(1);
      }),
    },
    {
      getOwnPropertySymbols: function (e) {
        return Zl.f(Vl(e));
      },
    },
  ),
  Ec) &&
    Ol(
      {
        target: "JSON",
        stat: !0,
        forced:
          !zl ||
          Ul(function () {
            var e = wc();
            return (
              "[null]" != Ec([e]) ||
              "{}" != Ec({ a: e }) ||
              "{}" != Ec(Object(e))
            );
          }),
      },
      {
        stringify: function (e, t, n) {
          for (var i, r = [e], a = 1; arguments.length > a; )
            r.push(arguments[a++]);
          if (((i = t), (ql(t) || void 0 !== e) && !Dc(e)))
            return (
              Bl(t) ||
                (t = function (e, t) {
                  if (
                    ("function" == typeof i && (t = i.call(this, e, t)), !Dc(t))
                  )
                    return t;
                }),
              (r[1] = t),
              Ec.apply(null, r)
            );
        },
      },
    );
  wc.prototype[gc] || ic(wc.prototype, gc, wc.prototype.valueOf),
    dc(wc, pc),
    (oc[vc] = !0);
  var Bc = Xt,
    qc = a,
    Wc = n,
    Vc = L,
    Kc = w,
    Yc = B.f,
    Xc = It,
    $c = Wc.Symbol;
  if (
    qc &&
    "function" == typeof $c &&
    (!("description" in $c.prototype) || void 0 !== $c().description)
  ) {
    var Gc = {},
      Jc = function () {
        var e =
            arguments.length < 1 || void 0 === arguments[0]
              ? void 0
              : String(arguments[0]),
          t = this instanceof Jc ? new $c(e) : void 0 === e ? $c() : $c(e);
        return "" === e && (Gc[t] = !0), t;
      };
    Xc(Jc, $c);
    var Qc = (Jc.prototype = $c.prototype);
    Qc.constructor = Jc;
    var Zc = Qc.toString,
      eu = "Symbol(test)" == String($c("test")),
      tu = /^Symbol\((.*)\)[^)]+$/;
    Yc(Qc, "description", {
      configurable: !0,
      get: function () {
        var e = Kc(this) ? this.valueOf() : this,
          t = Zc.call(e);
        if (Vc(Gc, e)) return "";
        var n = eu ? t.slice(7, -1) : t.replace(tu, "$1");
        return "" === n ? void 0 : n;
      },
    }),
      Bc({ global: !0, forced: !0 }, { Symbol: Jc });
  }
  var nu = W,
    iu = {},
    ru = r;
  function au(e, t) {
    return RegExp(e, t);
  }
  (iu.UNSUPPORTED_Y = ru(function () {
    var e = au("a", "y");
    return (e.lastIndex = 2), null != e.exec("abcd");
  })),
    (iu.BROKEN_CARET = ru(function () {
      var e = au("^r", "gy");
      return (e.lastIndex = 2), null != e.exec("str");
    }));
  var ou,
    su,
    lu = function () {
      var e = nu(this),
        t = "";
      return (
        e.global && (t += "g"),
        e.ignoreCase && (t += "i"),
        e.multiline && (t += "m"),
        e.dotAll && (t += "s"),
        e.unicode && (t += "u"),
        e.sticky && (t += "y"),
        t
      );
    },
    cu = iu,
    uu = he.exports,
    du = RegExp.prototype.exec,
    hu = uu("native-string-replace", String.prototype.replace),
    fu = du,
    vu =
      ((ou = /a/),
      (su = /b*/g),
      du.call(ou, "a"),
      du.call(su, "a"),
      0 !== ou.lastIndex || 0 !== su.lastIndex),
    pu = cu.UNSUPPORTED_Y || cu.BROKEN_CARET,
    gu = void 0 !== /()??/.exec("")[1];
  (vu || gu || pu) &&
    (fu = function (e) {
      var t,
        n,
        i,
        r,
        a = this,
        o = pu && a.sticky,
        s = lu.call(a),
        l = a.source,
        c = 0,
        u = e;
      return (
        o &&
          (-1 === (s = s.replace("y", "")).indexOf("g") && (s += "g"),
          (u = String(e).slice(a.lastIndex)),
          a.lastIndex > 0 &&
            (!a.multiline || (a.multiline && "\n" !== e[a.lastIndex - 1])) &&
            ((l = "(?: " + l + ")"), (u = " " + u), c++),
          (n = new RegExp("^(?:" + l + ")", s))),
        gu && (n = new RegExp("^" + l + "$(?!\\s)", s)),
        vu && (t = a.lastIndex),
        (i = du.call(o ? n : a, u)),
        o
          ? i
            ? ((i.input = i.input.slice(c)),
              (i[0] = i[0].slice(c)),
              (i.index = a.lastIndex),
              (a.lastIndex += i[0].length))
            : (a.lastIndex = 0)
          : vu && i && (a.lastIndex = a.global ? i.index + i[0].length : t),
        gu &&
          i &&
          i.length > 1 &&
          hu.call(i[0], n, function () {
            for (r = 1; r < arguments.length - 2; r++)
              void 0 === arguments[r] && (i[r] = void 0);
          }),
        i
      );
    });
  var mu = fu;
  Xt({ target: "RegExp", proto: !0, forced: /./.exec !== mu }, { exec: mu });
  var yu = Z.exports,
    bu = mu,
    wu = r,
    Eu = Dn,
    Su = Q,
    ku = Eu("species"),
    Au = RegExp.prototype,
    Ru = !wu(function () {
      var e = /./;
      return (
        (e.exec = function () {
          var e = [];
          return (e.groups = { a: "7" }), e;
        }),
        "7" !== "".replace(e, "$<a>")
      );
    }),
    xu = "$0" === "a".replace(/./, "$0"),
    Lu = Eu("replace"),
    Pu = !!/./[Lu] && "" === /./[Lu]("a", "$0"),
    Cu = !wu(function () {
      var e = /(?:)/,
        t = e.exec;
      e.exec = function () {
        return t.apply(this, arguments);
      };
      var n = "ab".split(e);
      return 2 !== n.length || "a" !== n[0] || "b" !== n[1];
    }),
    Nu = function (e, t, n, i) {
      var r = Eu(e),
        a = !wu(function () {
          var t = {};
          return (
            (t[r] = function () {
              return 7;
            }),
            7 != ""[e](t)
          );
        }),
        o =
          a &&
          !wu(function () {
            var t = !1,
              n = /a/;
            return (
              "split" === e &&
                (((n = {}).constructor = {}),
                (n.constructor[ku] = function () {
                  return n;
                }),
                (n.flags = ""),
                (n[r] = /./[r])),
              (n.exec = function () {
                return (t = !0), null;
              }),
              n[r](""),
              !t
            );
          });
      if (
        !a ||
        !o ||
        ("replace" === e && (!Ru || !xu || Pu)) ||
        ("split" === e && !Cu)
      ) {
        var s = /./[r],
          l = n(
            r,
            ""[e],
            function (e, t, n, i, r) {
              var o = t.exec;
              return o === bu || o === Au.exec
                ? a && !r
                  ? { done: !0, value: s.call(t, n, i) }
                  : { done: !0, value: e.call(n, t, i) }
                : { done: !1 };
            },
            {
              REPLACE_KEEPS_$0: xu,
              REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE: Pu,
            },
          ),
          c = l[0],
          u = l[1];
        yu(String.prototype, e, c),
          yu(
            Au,
            r,
            2 == t
              ? function (e, t) {
                  return u.call(e, this, t);
                }
              : function (e) {
                  return u.call(e, this);
                },
          );
      }
      i && Su(Au[r], "sham", !0);
    },
    Mu = cs.charAt,
    Iu = function (e, t, n) {
      return t + (n ? Mu(e, t).length : 1);
    },
    Tu = h,
    Ou = mu,
    Du = function (e, t) {
      var n = e.exec;
      if ("function" == typeof n) {
        var i = n.call(e, t);
        if ("object" != typeof i)
          throw TypeError(
            "RegExp exec method returned something other than an Object or null",
          );
        return i;
      }
      if ("RegExp" !== Tu(e))
        throw TypeError("RegExp#exec called on incompatible receiver");
      return Ou.call(e, t);
    },
    ju = W,
    Fu = at,
    zu = g,
    Hu = Iu,
    Uu = Du;
  Nu("match", 1, function (e, t, n) {
    return [
      function (t) {
        var n = zu(this),
          i = null == t ? void 0 : t[e];
        return void 0 !== i ? i.call(t, n) : new RegExp(t)[e](String(n));
      },
      function (e) {
        var i = n(t, e, this);
        if (i.done) return i.value;
        var r = ju(e),
          a = String(this);
        if (!r.global) return Uu(r, a);
        var o = r.unicode;
        r.lastIndex = 0;
        for (var s, l = [], c = 0; null !== (s = Uu(r, a)); ) {
          var u = String(s[0]);
          (l[c] = u),
            "" === u && (r.lastIndex = Hu(a, Fu(r.lastIndex), o)),
            c++;
        }
        return 0 === c ? null : l;
      },
    ];
  });
  var _u = Xt,
    Bu = ct,
    qu = nt,
    Wu = at,
    Vu = A,
    Ku = Hn,
    Yu = ni,
    Xu = Jn("splice"),
    $u = Math.max,
    Gu = Math.min,
    Ju = 9007199254740991,
    Qu = "Maximum allowed length exceeded";
  _u(
    { target: "Array", proto: !0, forced: !Xu },
    {
      splice: function (e, t) {
        var n,
          i,
          r,
          a,
          o,
          s,
          l = Vu(this),
          c = Wu(l.length),
          u = Bu(e, c),
          d = arguments.length;
        if (
          (0 === d
            ? (n = i = 0)
            : 1 === d
            ? ((n = 0), (i = c - u))
            : ((n = d - 2), (i = Gu($u(qu(t), 0), c - u))),
          c + n - i > Ju)
        )
          throw TypeError(Qu);
        for (r = Ku(l, i), a = 0; a < i; a++)
          (o = u + a) in l && Yu(r, a, l[o]);
        if (((r.length = i), n < i)) {
          for (a = u; a < c - i; a++)
            (s = a + n), (o = a + i) in l ? (l[s] = l[o]) : delete l[s];
          for (a = c; a > c - i + n; a--) delete l[a - 1];
        } else if (n > i)
          for (a = c - i; a > u; a--)
            (s = a + n - 1), (o = a + i - 1) in l ? (l[s] = l[o]) : delete l[s];
        for (a = 0; a < n; a++) l[a + u] = arguments[a + 2];
        return (l.length = c - i + n), r;
      },
    },
  );
  var Zu = w,
    ed = h,
    td = Dn("match"),
    nd = Nu,
    id = function (e) {
      var t;
      return Zu(e) && (void 0 !== (t = e[td]) ? !!t : "RegExp" == ed(e));
    },
    rd = W,
    ad = g,
    od = Nr,
    sd = Iu,
    ld = at,
    cd = Du,
    ud = mu,
    dd = iu.UNSUPPORTED_Y,
    hd = [].push,
    fd = Math.min,
    vd = 4294967295;
  nd(
    "split",
    2,
    function (e, t, n) {
      var i;
      return (
        (i =
          "c" == "abbc".split(/(b)*/)[1] ||
          4 != "test".split(/(?:)/, -1).length ||
          2 != "ab".split(/(?:ab)*/).length ||
          4 != ".".split(/(.?)(.?)/).length ||
          ".".split(/()()/).length > 1 ||
          "".split(/.?/).length
            ? function (e, n) {
                var i = String(ad(this)),
                  r = void 0 === n ? vd : n >>> 0;
                if (0 === r) return [];
                if (void 0 === e) return [i];
                if (!id(e)) return t.call(i, e, r);
                for (
                  var a,
                    o,
                    s,
                    l = [],
                    c =
                      (e.ignoreCase ? "i" : "") +
                      (e.multiline ? "m" : "") +
                      (e.unicode ? "u" : "") +
                      (e.sticky ? "y" : ""),
                    u = 0,
                    d = new RegExp(e.source, c + "g");
                  (a = ud.call(d, i)) &&
                  !(
                    (o = d.lastIndex) > u &&
                    (l.push(i.slice(u, a.index)),
                    a.length > 1 &&
                      a.index < i.length &&
                      hd.apply(l, a.slice(1)),
                    (s = a[0].length),
                    (u = o),
                    l.length >= r)
                  );

                )
                  d.lastIndex === a.index && d.lastIndex++;
                return (
                  u === i.length
                    ? (!s && d.test("")) || l.push("")
                    : l.push(i.slice(u)),
                  l.length > r ? l.slice(0, r) : l
                );
              }
            : "0".split(void 0, 0).length
            ? function (e, n) {
                return void 0 === e && 0 === n ? [] : t.call(this, e, n);
              }
            : t),
        [
          function (t, n) {
            var r = ad(this),
              a = null == t ? void 0 : t[e];
            return void 0 !== a ? a.call(t, r, n) : i.call(String(r), t, n);
          },
          function (e, r) {
            var a = n(i, e, this, r, i !== t);
            if (a.done) return a.value;
            var o = rd(e),
              s = String(this),
              l = od(o, RegExp),
              c = o.unicode,
              u =
                (o.ignoreCase ? "i" : "") +
                (o.multiline ? "m" : "") +
                (o.unicode ? "u" : "") +
                (dd ? "g" : "y"),
              d = new l(dd ? "^(?:" + o.source + ")" : o, u),
              h = void 0 === r ? vd : r >>> 0;
            if (0 === h) return [];
            if (0 === s.length) return null === cd(d, s) ? [s] : [];
            for (var f = 0, v = 0, p = []; v < s.length; ) {
              d.lastIndex = dd ? 0 : v;
              var g,
                m = cd(d, dd ? s.slice(v) : s);
              if (
                null === m ||
                (g = fd(ld(d.lastIndex + (dd ? v : 0)), s.length)) === f
              )
                v = sd(s, v, c);
              else {
                if ((p.push(s.slice(f, v)), p.length === h)) return p;
                for (var y = 1; y <= m.length - 1; y++)
                  if ((p.push(m[y]), p.length === h)) return p;
                v = f = g;
              }
            }
            return p.push(s.slice(f)), p;
          },
        ]
      );
    },
    dd,
  );
  var pd = w,
    gd = Ki,
    md = a,
    yd = n,
    bd = Ut,
    wd = Z.exports,
    Ed = L,
    Sd = h,
    kd = function (e, t, n) {
      var i, r;
      return (
        gd &&
          "function" == typeof (i = t.constructor) &&
          i !== n &&
          pd((r = i.prototype)) &&
          r !== n.prototype &&
          gd(e, r),
        e
      );
    },
    Ad = S,
    Rd = r,
    xd = Hs,
    Ld = Ze.f,
    Pd = i.f,
    Cd = B.f,
    Nd = bl.trim,
    Md = "Number",
    Id = yd.Number,
    Td = Id.prototype,
    Od = Sd(xd(Td)) == Md,
    Dd = function (e) {
      var t,
        n,
        i,
        r,
        a,
        o,
        s,
        l,
        c = Ad(e, !1);
      if ("string" == typeof c && c.length > 2)
        if (43 === (t = (c = Nd(c)).charCodeAt(0)) || 45 === t) {
          if (88 === (n = c.charCodeAt(2)) || 120 === n) return NaN;
        } else if (48 === t) {
          switch (c.charCodeAt(1)) {
            case 66:
            case 98:
              (i = 2), (r = 49);
              break;
            case 79:
            case 111:
              (i = 8), (r = 55);
              break;
            default:
              return +c;
          }
          for (o = (a = c.slice(2)).length, s = 0; s < o; s++)
            if ((l = a.charCodeAt(s)) < 48 || l > r) return NaN;
          return parseInt(a, i);
        }
      return +c;
    };
  if (bd(Md, !Id(" 0o1") || !Id("0b1") || Id("+0x1"))) {
    for (
      var jd,
        Fd = function (e) {
          var t = arguments.length < 1 ? 0 : e,
            n = this;
          return n instanceof Fd &&
            (Od
              ? Rd(function () {
                  Td.valueOf.call(n);
                })
              : Sd(n) != Md)
            ? kd(new Id(Dd(t)), n, Fd)
            : Dd(t);
        },
        zd = md
          ? Ld(Id)
          : "MAX_VALUE,MIN_VALUE,NaN,NEGATIVE_INFINITY,POSITIVE_INFINITY,EPSILON,isFinite,isInteger,isNaN,isSafeInteger,MAX_SAFE_INTEGER,MIN_SAFE_INTEGER,parseFloat,parseInt,isInteger,fromString,range".split(
              ",",
            ),
        Hd = 0;
      zd.length > Hd;
      Hd++
    )
      Ed(Id, (jd = zd[Hd])) && !Ed(Fd, jd) && Cd(Fd, jd, Pd(Id, jd));
    (Fd.prototype = Td), (Td.constructor = Fd), wd(yd, Md, Fd);
  }
  var Ud = a,
    _d = B.f,
    Bd = Function.prototype,
    qd = Bd.toString,
    Wd = /^\s*function ([^ (]*)/,
    Vd = "name";
  Ud &&
    !(Vd in Bd) &&
    _d(Bd, Vd, {
      configurable: !0,
      get: function () {
        try {
          return qd.call(this).match(Wd)[1];
        } catch (e) {
          return "";
        }
      },
    });
  var Kd = Xt,
    Yd = b,
    Xd = [].join,
    $d = p != Object,
    Gd = zo("join", ",");
  Kd(
    { target: "Array", proto: !0, forced: $d || !Gd },
    {
      join: function (e) {
        return Xd.call(Yd(this), void 0 === e ? "," : e);
      },
    },
  );
  var Jd = A,
    Qd = Math.floor,
    Zd = "".replace,
    eh = /\$([$&'`]|\d{1,2}|<[^>]*>)/g,
    th = /\$([$&'`]|\d{1,2})/g,
    nh = Nu,
    ih = W,
    rh = at,
    ah = nt,
    oh = g,
    sh = Iu,
    lh = function (e, t, n, i, r, a) {
      var o = n + e.length,
        s = i.length,
        l = th;
      return (
        void 0 !== r && ((r = Jd(r)), (l = eh)),
        Zd.call(a, l, function (a, l) {
          var c;
          switch (l.charAt(0)) {
            case "$":
              return "$";
            case "&":
              return e;
            case "`":
              return t.slice(0, n);
            case "'":
              return t.slice(o);
            case "<":
              c = r[l.slice(1, -1)];
              break;
            default:
              var u = +l;
              if (0 === u) return a;
              if (u > s) {
                var d = Qd(u / 10);
                return 0 === d
                  ? a
                  : d <= s
                  ? void 0 === i[d - 1]
                    ? l.charAt(1)
                    : i[d - 1] + l.charAt(1)
                  : a;
              }
              c = i[u - 1];
          }
          return void 0 === c ? "" : c;
        })
      );
    },
    ch = Du,
    uh = Math.max,
    dh = Math.min;
  nh("replace", 2, function (e, t, n, i) {
    var r = i.REGEXP_REPLACE_SUBSTITUTES_UNDEFINED_CAPTURE,
      a = i.REPLACE_KEEPS_$0,
      o = r ? "$" : "$0";
    return [
      function (n, i) {
        var r = oh(this),
          a = null == n ? void 0 : n[e];
        return void 0 !== a ? a.call(n, r, i) : t.call(String(r), n, i);
      },
      function (e, i) {
        if ((!r && a) || ("string" == typeof i && -1 === i.indexOf(o))) {
          var s = n(t, e, this, i);
          if (s.done) return s.value;
        }
        var l = ih(e),
          c = String(this),
          u = "function" == typeof i;
        u || (i = String(i));
        var d = l.global;
        if (d) {
          var h = l.unicode;
          l.lastIndex = 0;
        }
        for (var f = []; ; ) {
          var v = ch(l, c);
          if (null === v) break;
          if ((f.push(v), !d)) break;
          "" === String(v[0]) && (l.lastIndex = sh(c, rh(l.lastIndex), h));
        }
        for (var p, g = "", m = 0, y = 0; y < f.length; y++) {
          v = f[y];
          for (
            var b = String(v[0]),
              w = uh(dh(ah(v.index), c.length), 0),
              E = [],
              S = 1;
            S < v.length;
            S++
          )
            E.push(void 0 === (p = v[S]) ? p : String(p));
          var k = v.groups;
          if (u) {
            var A = [b].concat(E, w, c);
            void 0 !== k && A.push(k);
            var R = String(i.apply(void 0, A));
          } else R = lh(b, c, w, E, k, i);
          w >= m && ((g += c.slice(m, w) + R), (m = w + b.length));
        }
        return g + c.slice(m);
      },
    ];
  });
  var hh =
      Object.is ||
      function (e, t) {
        return e === t ? 0 !== e || 1 / e == 1 / t : e != e && t != t;
      },
    fh = W,
    vh = g,
    ph = hh,
    gh = Du;
  Nu("search", 1, function (e, t, n) {
    return [
      function (t) {
        var n = vh(this),
          i = null == t ? void 0 : t[e];
        return void 0 !== i ? i.call(t, n) : new RegExp(t)[e](String(n));
      },
      function (e) {
        var i = n(t, e, this);
        if (i.done) return i.value;
        var r = fh(e),
          a = String(this),
          o = r.lastIndex;
        ph(o, 0) || (r.lastIndex = 0);
        var s = gh(r, a);
        return (
          ph(r.lastIndex, o) || (r.lastIndex = o), null === s ? -1 : s.index
        );
      },
    ];
  });
  var mh = function (e, t) {
      for (var n in t) e[n] = t[n];
      return e;
    },
    yh = function (e, t) {
      return Array.from(e.querySelectorAll(t));
    },
    bh = function (e, t, n) {
      n ? e.classList.add(t) : e.classList.remove(t);
    },
    wh = function (e) {
      if ("string" == typeof e) {
        if ("null" === e) return null;
        if ("true" === e) return !0;
        if ("false" === e) return !1;
        if (e.match(/^-?[\d\.]+$/)) return parseFloat(e);
      }
      return e;
    },
    Eh = function (e, t) {
      e.style.transform = t;
    },
    Sh = function (e, t) {
      var n = e.matches || e.matchesSelector || e.msMatchesSelector;
      return !(!n || !n.call(e, t));
    },
    kh = function (e, t) {
      if ("function" == typeof e.closest) return e.closest(t);
      for (; e; ) {
        if (Sh(e, t)) return e;
        e = e.parentNode;
      }
      return null;
    },
    Ah = function (e, t, n) {
      for (
        var i =
            arguments.length > 3 && void 0 !== arguments[3] ? arguments[3] : "",
          r = e.querySelectorAll("." + n),
          a = 0;
        a < r.length;
        a++
      ) {
        var o = r[a];
        if (o.parentNode === e) return o;
      }
      var s = document.createElement(t);
      return (s.className = n), (s.innerHTML = i), e.appendChild(s), s;
    },
    Rh = function (e) {
      var t = document.createElement("style");
      return (
        (t.type = "text/css"),
        e &&
          e.length > 0 &&
          (t.styleSheet
            ? (t.styleSheet.cssText = e)
            : t.appendChild(document.createTextNode(e))),
        document.head.appendChild(t),
        t
      );
    },
    xh = function () {
      var e = {};
      for (var t in (location.search.replace(
        /[A-Z0-9]+?=([\w\.%-]*)/gi,
        function (t) {
          e[t.split("=").shift()] = t.split("=").pop();
        },
      ),
      e)) {
        var n = e[t];
        e[t] = wh(unescape(n));
      }
      return void 0 !== e.dependencies && delete e.dependencies, e;
    },
    Lh = function (e) {
      var t =
        arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : 0;
      if (e) {
        var n,
          i = e.style.height;
        return (
          (e.style.height = "0px"),
          (e.parentNode.style.height = "auto"),
          (n = t - e.parentNode.offsetHeight),
          (e.style.height = i + "px"),
          e.parentNode.style.removeProperty("height"),
          n
        );
      }
      return t;
    },
    Ph = {
      mp4: "video/mp4",
      m4a: "video/mp4",
      ogv: "video/ogg",
      mpeg: "video/mpeg",
      webm: "video/webm",
    },
    Ch = function () {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : "";
      return Ph[e.split(".").pop()];
    },
    Nh = navigator.userAgent,
    Mh =
      /(iphone|ipod|ipad|android)/gi.test(Nh) ||
      ("MacIntel" === navigator.platform && navigator.maxTouchPoints > 1);
  /chrome/i.test(Nh) && /edge/i.test(Nh);
  var Ih = /android/gi.test(Nh),
    Th = {};
  Object.defineProperty(Th, "__esModule", { value: !0 });
  var Oh =
      Object.assign ||
      function (e) {
        for (var t = 1; t < arguments.length; t++) {
          var n = arguments[t];
          for (var i in n)
            Object.prototype.hasOwnProperty.call(n, i) && (e[i] = n[i]);
        }
        return e;
      },
    Dh = (Th.default = (function (e) {
      if (e) {
        var t = function (e) {
            return [].slice.call(e);
          },
          n = 0,
          i = 1,
          r = 2,
          a = 3,
          o = [],
          s = null,
          l =
            "requestAnimationFrame" in e
              ? function () {
                  e.cancelAnimationFrame(s),
                    (s = e.requestAnimationFrame(function () {
                      return u(
                        o.filter(function (e) {
                          return e.dirty && e.active;
                        }),
                      );
                    }));
                }
              : function () {},
          c = function (e) {
            return function () {
              o.forEach(function (t) {
                return (t.dirty = e);
              }),
                l();
            };
          },
          u = function (e) {
            e
              .filter(function (e) {
                return !e.styleComputed;
              })
              .forEach(function (e) {
                e.styleComputed = v(e);
              }),
              e.filter(p).forEach(g);
            var t = e.filter(f);
            t.forEach(h),
              t.forEach(function (e) {
                g(e), d(e);
              }),
              t.forEach(m);
          },
          d = function (e) {
            return (e.dirty = n);
          },
          h = function (e) {
            (e.availableWidth = e.element.parentNode.clientWidth),
              (e.currentWidth = e.element.scrollWidth),
              (e.previousFontSize = e.currentFontSize),
              (e.currentFontSize = Math.min(
                Math.max(
                  e.minSize,
                  (e.availableWidth / e.currentWidth) * e.previousFontSize,
                ),
                e.maxSize,
              )),
              (e.whiteSpace =
                e.multiLine && e.currentFontSize === e.minSize
                  ? "normal"
                  : "nowrap");
          },
          f = function (e) {
            return (
              e.dirty !== r ||
              (e.dirty === r &&
                e.element.parentNode.clientWidth !== e.availableWidth)
            );
          },
          v = function (t) {
            var n = e.getComputedStyle(t.element, null);
            (t.currentFontSize = parseFloat(n.getPropertyValue("font-size"))),
              (t.display = n.getPropertyValue("display")),
              (t.whiteSpace = n.getPropertyValue("white-space"));
          },
          p = function (e) {
            var t = !1;
            return (
              !e.preStyleTestCompleted &&
              (/inline-/.test(e.display) ||
                ((t = !0), (e.display = "inline-block")),
              "nowrap" !== e.whiteSpace &&
                ((t = !0), (e.whiteSpace = "nowrap")),
              (e.preStyleTestCompleted = !0),
              t)
            );
          },
          g = function (e) {
            (e.element.style.whiteSpace = e.whiteSpace),
              (e.element.style.display = e.display),
              (e.element.style.fontSize = e.currentFontSize + "px");
          },
          m = function (e) {
            e.element.dispatchEvent(
              new CustomEvent("fit", {
                detail: {
                  oldValue: e.previousFontSize,
                  newValue: e.currentFontSize,
                  scaleFactor: e.currentFontSize / e.previousFontSize,
                },
              }),
            );
          },
          y = function (e, t) {
            return function () {
              (e.dirty = t), e.active && l();
            };
          },
          b = function (e) {
            return function () {
              (o = o.filter(function (t) {
                return t.element !== e.element;
              })),
                e.observeMutations && e.observer.disconnect(),
                (e.element.style.whiteSpace = e.originalStyle.whiteSpace),
                (e.element.style.display = e.originalStyle.display),
                (e.element.style.fontSize = e.originalStyle.fontSize);
            };
          },
          w = function (e) {
            return function () {
              e.active || ((e.active = !0), l());
            };
          },
          E = function (e) {
            return function () {
              return (e.active = !1);
            };
          },
          S = function (e) {
            e.observeMutations &&
              ((e.observer = new MutationObserver(y(e, i))),
              e.observer.observe(e.element, e.observeMutations));
          },
          k = {
            minSize: 16,
            maxSize: 512,
            multiLine: !0,
            observeMutations: "MutationObserver" in e && {
              subtree: !0,
              childList: !0,
              characterData: !0,
            },
          },
          A = null,
          R = function () {
            e.clearTimeout(A), (A = e.setTimeout(c(r), P.observeWindowDelay));
          },
          x = ["resize", "orientationchange"];
        return (
          Object.defineProperty(P, "observeWindow", {
            set: function (t) {
              var n = (t ? "add" : "remove") + "EventListener";
              x.forEach(function (t) {
                e[n](t, R);
              });
            },
          }),
          (P.observeWindow = !0),
          (P.observeWindowDelay = 100),
          (P.fitAll = c(a)),
          P
        );
      }
      function L(e, t) {
        var n = Oh({}, k, t),
          i = e.map(function (e) {
            var t = Oh({}, n, { element: e, active: !0 });
            return (
              (function (e) {
                (e.originalStyle = {
                  whiteSpace: e.element.style.whiteSpace,
                  display: e.element.style.display,
                  fontSize: e.element.style.fontSize,
                }),
                  S(e),
                  (e.newbie = !0),
                  (e.dirty = !0),
                  o.push(e);
              })(t),
              {
                element: e,
                fit: y(t, a),
                unfreeze: w(t),
                freeze: E(t),
                unsubscribe: b(t),
              }
            );
          });
        return l(), i;
      }
      function P(e) {
        var n =
          arguments.length > 1 && void 0 !== arguments[1] ? arguments[1] : {};
        return "string" == typeof e
          ? L(t(document.querySelectorAll(e)), n)
          : L([e], n)[0];
      }
    })("undefined" == typeof window ? null : window)),
    jh = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.startEmbeddedIframe = this.startEmbeddedIframe.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "shouldPreload",
            value: function (e) {
              var t = this.Reveal.getConfig().preloadIframes;
              return (
                "boolean" != typeof t && (t = e.hasAttribute("data-preload")), t
              );
            },
          },
          {
            key: "load",
            value: function (e) {
              var t = this,
                n =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : {};
              (e.style.display = this.Reveal.getConfig().display),
                yh(
                  e,
                  "img[data-src], video[data-src], audio[data-src], iframe[data-src]",
                ).forEach(function (e) {
                  ("IFRAME" !== e.tagName || t.shouldPreload(e)) &&
                    (e.setAttribute("src", e.getAttribute("data-src")),
                    e.setAttribute("data-lazy-loaded", ""),
                    e.removeAttribute("data-src"));
                }),
                yh(e, "video, audio").forEach(function (e) {
                  var t = 0;
                  yh(e, "source[data-src]").forEach(function (e) {
                    e.setAttribute("src", e.getAttribute("data-src")),
                      e.removeAttribute("data-src"),
                      e.setAttribute("data-lazy-loaded", ""),
                      (t += 1);
                  }),
                    Mh &&
                      "VIDEO" === e.tagName &&
                      e.setAttribute("playsinline", ""),
                    t > 0 && e.load();
                });
              var i = e.slideBackgroundElement;
              if (i) {
                i.style.display = "block";
                var r = e.slideBackgroundContentElement,
                  a = e.getAttribute("data-background-iframe");
                if (!1 === i.hasAttribute("data-loaded")) {
                  i.setAttribute("data-loaded", "true");
                  var o = e.getAttribute("data-background-image"),
                    s = e.getAttribute("data-background-video"),
                    l = e.hasAttribute("data-background-video-loop"),
                    c = e.hasAttribute("data-background-video-muted");
                  if (o)
                    /^data:/.test(o.trim())
                      ? (r.style.backgroundImage = "url(".concat(o.trim(), ")"))
                      : (r.style.backgroundImage = o
                          .split(",")
                          .map(function (e) {
                            return "url(".concat(encodeURI(e.trim()), ")");
                          })
                          .join(","));
                  else if (s && !this.Reveal.isSpeakerNotes()) {
                    var u = document.createElement("video");
                    l && u.setAttribute("loop", ""),
                      c && (u.muted = !0),
                      Mh && ((u.muted = !0), u.setAttribute("playsinline", "")),
                      s.split(",").forEach(function (e) {
                        var t = Ch(e);
                        u.innerHTML += t
                          ? '<source src="'
                              .concat(e, '" type="')
                              .concat(t, '">')
                          : '<source src="'.concat(e, '">');
                      }),
                      r.appendChild(u);
                  } else if (a && !0 !== n.excludeIframes) {
                    var d = document.createElement("iframe");
                    d.setAttribute("allowfullscreen", ""),
                      d.setAttribute("mozallowfullscreen", ""),
                      d.setAttribute("webkitallowfullscreen", ""),
                      d.setAttribute("allow", "autoplay"),
                      d.setAttribute("data-src", a),
                      (d.style.width = "100%"),
                      (d.style.height = "100%"),
                      (d.style.maxHeight = "100%"),
                      (d.style.maxWidth = "100%"),
                      r.appendChild(d);
                  }
                }
                var h = r.querySelector("iframe[data-src]");
                h &&
                  this.shouldPreload(i) &&
                  !/autoplay=(1|true|yes)/gi.test(a) &&
                  h.getAttribute("src") !== a &&
                  h.setAttribute("src", a);
              }
              this.layout(e);
            },
          },
          {
            key: "layout",
            value: function (e) {
              var t = this;
              Array.from(e.querySelectorAll(".r-fit-text")).forEach(
                function (e) {
                  Dh(e, {
                    minSize: 24,
                    maxSize: 0.8 * t.Reveal.getConfig().height,
                    observeMutations: !1,
                    observeWindow: !1,
                  });
                },
              );
            },
          },
          {
            key: "unload",
            value: function (e) {
              e.style.display = "none";
              var t = this.Reveal.getSlideBackground(e);
              t &&
                ((t.style.display = "none"),
                yh(t, "iframe[src]").forEach(function (e) {
                  e.removeAttribute("src");
                })),
                yh(
                  e,
                  "video[data-lazy-loaded][src], audio[data-lazy-loaded][src], iframe[data-lazy-loaded][src]",
                ).forEach(function (e) {
                  e.setAttribute("data-src", e.getAttribute("src")),
                    e.removeAttribute("src");
                }),
                yh(
                  e,
                  "video[data-lazy-loaded] source[src], audio source[src]",
                ).forEach(function (e) {
                  e.setAttribute("data-src", e.getAttribute("src")),
                    e.removeAttribute("src");
                });
            },
          },
          {
            key: "formatEmbeddedContent",
            value: function () {
              var e = this,
                t = function (t, n, i) {
                  yh(
                    e.Reveal.getSlidesElement(),
                    "iframe[" + t + '*="' + n + '"]',
                  ).forEach(function (e) {
                    var n = e.getAttribute(t);
                    n &&
                      -1 === n.indexOf(i) &&
                      e.setAttribute(t, n + (/\?/.test(n) ? "&" : "?") + i);
                  });
                };
              t("src", "youtube.com/embed/", "enablejsapi=1"),
                t("data-src", "youtube.com/embed/", "enablejsapi=1"),
                t("src", "player.vimeo.com/", "api=1"),
                t("data-src", "player.vimeo.com/", "api=1");
            },
          },
          {
            key: "startEmbeddedContent",
            value: function (e) {
              var t = this;
              e &&
                !this.Reveal.isSpeakerNotes() &&
                (yh(e, 'img[src$=".gif"]').forEach(function (e) {
                  e.setAttribute("src", e.getAttribute("src"));
                }),
                yh(e, "video, audio").forEach(function (e) {
                  if (!kh(e, ".fragment") || kh(e, ".fragment.visible")) {
                    var n = t.Reveal.getConfig().autoPlayMedia;
                    if (
                      ("boolean" != typeof n &&
                        (n =
                          e.hasAttribute("data-autoplay") ||
                          !!kh(e, ".slide-background")),
                      n && "function" == typeof e.play)
                    )
                      if (e.readyState > 1) t.startEmbeddedMedia({ target: e });
                      else if (Mh) {
                        var i = e.play();
                        i &&
                          "function" == typeof i.catch &&
                          !1 === e.controls &&
                          i.catch(function () {
                            (e.controls = !0),
                              e.addEventListener("play", function () {
                                e.controls = !1;
                              });
                          });
                      } else
                        e.removeEventListener(
                          "loadeddata",
                          t.startEmbeddedMedia,
                        ),
                          e.addEventListener(
                            "loadeddata",
                            t.startEmbeddedMedia,
                          );
                  }
                }),
                yh(e, "iframe[src]").forEach(function (e) {
                  (kh(e, ".fragment") && !kh(e, ".fragment.visible")) ||
                    t.startEmbeddedIframe({ target: e });
                }),
                yh(e, "iframe[data-src]").forEach(function (e) {
                  (kh(e, ".fragment") && !kh(e, ".fragment.visible")) ||
                    (e.getAttribute("src") !== e.getAttribute("data-src") &&
                      (e.removeEventListener("load", t.startEmbeddedIframe),
                      e.addEventListener("load", t.startEmbeddedIframe),
                      e.setAttribute("src", e.getAttribute("data-src"))));
                }));
            },
          },
          {
            key: "startEmbeddedMedia",
            value: function (e) {
              var t = !!kh(e.target, "html"),
                n = !!kh(e.target, ".present");
              t && n && ((e.target.currentTime = 0), e.target.play()),
                e.target.removeEventListener(
                  "loadeddata",
                  this.startEmbeddedMedia,
                );
            },
          },
          {
            key: "startEmbeddedIframe",
            value: function (e) {
              var t = e.target;
              if (t && t.contentWindow) {
                var n = !!kh(e.target, "html"),
                  i = !!kh(e.target, ".present");
                if (n && i) {
                  var r = this.Reveal.getConfig().autoPlayMedia;
                  "boolean" != typeof r &&
                    (r =
                      t.hasAttribute("data-autoplay") ||
                      !!kh(t, ".slide-background")),
                    /youtube\.com\/embed\//.test(t.getAttribute("src")) && r
                      ? t.contentWindow.postMessage(
                          '{"event":"command","func":"playVideo","args":""}',
                          "*",
                        )
                      : /player\.vimeo\.com\//.test(t.getAttribute("src")) && r
                      ? t.contentWindow.postMessage('{"method":"play"}', "*")
                      : t.contentWindow.postMessage("slide:start", "*");
                }
              }
            },
          },
          {
            key: "stopEmbeddedContent",
            value: function (e) {
              var t = this,
                n =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : {};
              (n = mh({ unloadIframes: !0 }, n)),
                e &&
                  e.parentNode &&
                  (yh(e, "video, audio").forEach(function (e) {
                    e.hasAttribute("data-ignore") ||
                      "function" != typeof e.pause ||
                      (e.setAttribute("data-paused-by-reveal", ""), e.pause());
                  }),
                  yh(e, "iframe").forEach(function (e) {
                    e.contentWindow &&
                      e.contentWindow.postMessage("slide:stop", "*"),
                      e.removeEventListener("load", t.startEmbeddedIframe);
                  }),
                  yh(e, 'iframe[src*="youtube.com/embed/"]').forEach(
                    function (e) {
                      !e.hasAttribute("data-ignore") &&
                        e.contentWindow &&
                        "function" == typeof e.contentWindow.postMessage &&
                        e.contentWindow.postMessage(
                          '{"event":"command","func":"pauseVideo","args":""}',
                          "*",
                        );
                    },
                  ),
                  yh(e, 'iframe[src*="player.vimeo.com/"]').forEach(
                    function (e) {
                      !e.hasAttribute("data-ignore") &&
                        e.contentWindow &&
                        "function" == typeof e.contentWindow.postMessage &&
                        e.contentWindow.postMessage('{"method":"pause"}', "*");
                    },
                  ),
                  !0 === n.unloadIframes &&
                    yh(e, "iframe[data-src]").forEach(function (e) {
                      e.setAttribute("src", "about:blank"),
                        e.removeAttribute("src");
                    }));
            },
          },
        ]),
        e
      );
    })(),
    Fh = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      return (
        Ri(e, [
          {
            key: "render",
            value: function () {
              (this.element = document.createElement("div")),
                (this.element.className = "slide-number"),
                this.Reveal.getRevealElement().appendChild(this.element);
            },
          },
          {
            key: "configure",
            value: function (e, t) {
              var n = "none";
              e.slideNumber &&
                !this.Reveal.isPrintingPDF() &&
                ("all" === e.showSlideNumber ||
                  ("speaker" === e.showSlideNumber &&
                    this.Reveal.isSpeakerNotes())) &&
                (n = "block"),
                (this.element.style.display = n);
            },
          },
          {
            key: "update",
            value: function () {
              this.Reveal.getConfig().slideNumber &&
                this.element &&
                (this.element.innerHTML = this.getSlideNumber());
            },
          },
          {
            key: "getSlideNumber",
            value: function () {
              var e,
                t =
                  arguments.length > 0 && void 0 !== arguments[0]
                    ? arguments[0]
                    : this.Reveal.getCurrentSlide(),
                n = this.Reveal.getConfig(),
                i = "h.v";
              if ("function" == typeof n.slideNumber) e = n.slideNumber(t);
              else {
                "string" == typeof n.slideNumber && (i = n.slideNumber),
                  /c/.test(i) ||
                    1 !== this.Reveal.getHorizontalSlides().length ||
                    (i = "c");
                var r = t && "uncounted" === t.dataset.visibility ? 0 : 1;
                switch (((e = []), i)) {
                  case "c":
                    e.push(this.Reveal.getSlidePastCount(t) + r);
                    break;
                  case "c/t":
                    e.push(
                      this.Reveal.getSlidePastCount(t) + r,
                      "/",
                      this.Reveal.getTotalSlides(),
                    );
                    break;
                  default:
                    var a = this.Reveal.getIndices(t);
                    e.push(a.h + r);
                    var o = "h/v" === i ? "/" : ".";
                    this.Reveal.isVerticalSlide(t) && e.push(o, a.v + 1);
                }
              }
              var s = "#" + this.Reveal.location.getHash(t);
              return this.formatNumber(e[0], e[1], e[2], s);
            },
          },
          {
            key: "formatNumber",
            value: function (e, t, n) {
              var i =
                arguments.length > 3 && void 0 !== arguments[3]
                  ? arguments[3]
                  : "#" + this.Reveal.location.getHash();
              return "number" != typeof n || isNaN(n)
                ? '<a href="'
                    .concat(i, '">\n\t\t\t\t\t<span class="slide-number-a">')
                    .concat(e, "</span>\n\t\t\t\t\t</a>")
                : '<a href="'
                    .concat(i, '">\n\t\t\t\t\t<span class="slide-number-a">')
                    .concat(
                      e,
                      '</span>\n\t\t\t\t\t<span class="slide-number-delimiter">',
                    )
                    .concat(
                      t,
                      '</span>\n\t\t\t\t\t<span class="slide-number-b">',
                    )
                    .concat(n, "</span>\n\t\t\t\t\t</a>");
            },
          },
          {
            key: "destroy",
            value: function () {
              this.element.remove();
            },
          },
        ]),
        e
      );
    })(),
    zh = Xt,
    Hh = w,
    Uh = pn,
    _h = ct,
    Bh = at,
    qh = b,
    Wh = ni,
    Vh = Dn,
    Kh = Jn("slice"),
    Yh = Vh("species"),
    Xh = [].slice,
    $h = Math.max;
  zh(
    { target: "Array", proto: !0, forced: !Kh },
    {
      slice: function (e, t) {
        var n,
          i,
          r,
          a = qh(this),
          o = Bh(a.length),
          s = _h(e, o),
          l = _h(void 0 === t ? o : t, o);
        if (
          Uh(a) &&
          ("function" != typeof (n = a.constructor) ||
          (n !== Array && !Uh(n.prototype))
            ? Hh(n) && null === (n = n[Yh]) && (n = void 0)
            : (n = void 0),
          n === Array || void 0 === n)
        )
          return Xh.call(a, s, l);
        for (
          i = new (void 0 === n ? Array : n)($h(l - s, 0)), r = 0;
          s < l;
          s++, r++
        )
          s in a && Wh(i, r, a[s]);
        return (i.length = r), i;
      },
    },
  );
  var Gh = function (e) {
      var t = e.match(/^#([0-9a-f]{3})$/i);
      if (t && t[1])
        return (
          (t = t[1]),
          {
            r: 17 * parseInt(t.charAt(0), 16),
            g: 17 * parseInt(t.charAt(1), 16),
            b: 17 * parseInt(t.charAt(2), 16),
          }
        );
      var n = e.match(/^#([0-9a-f]{6})$/i);
      if (n && n[1])
        return (
          (n = n[1]),
          {
            r: parseInt(n.slice(0, 2), 16),
            g: parseInt(n.slice(2, 4), 16),
            b: parseInt(n.slice(4, 6), 16),
          }
        );
      var i = e.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i);
      if (i)
        return {
          r: parseInt(i[1], 10),
          g: parseInt(i[2], 10),
          b: parseInt(i[3], 10),
        };
      var r = e.match(
        /^rgba\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\,\s*([\d]+|[\d]*.[\d]+)\s*\)$/i,
      );
      return r
        ? {
            r: parseInt(r[1], 10),
            g: parseInt(r[2], 10),
            b: parseInt(r[3], 10),
            a: parseFloat(r[4]),
          }
        : null;
    },
    Jh = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      return (
        Ri(e, [
          {
            key: "render",
            value: function () {
              (this.element = document.createElement("div")),
                (this.element.className = "backgrounds"),
                this.Reveal.getRevealElement().appendChild(this.element);
            },
          },
          {
            key: "create",
            value: function () {
              var e = this;
              (this.element.innerHTML = ""),
                this.element.classList.add("no-transition"),
                this.Reveal.getHorizontalSlides().forEach(function (t) {
                  var n = e.createBackground(t, e.element);
                  yh(t, "section").forEach(function (t) {
                    e.createBackground(t, n), n.classList.add("stack");
                  });
                }),
                this.Reveal.getConfig().parallaxBackgroundImage
                  ? ((this.element.style.backgroundImage =
                      'url("' +
                      this.Reveal.getConfig().parallaxBackgroundImage +
                      '")'),
                    (this.element.style.backgroundSize =
                      this.Reveal.getConfig().parallaxBackgroundSize),
                    (this.element.style.backgroundRepeat =
                      this.Reveal.getConfig().parallaxBackgroundRepeat),
                    (this.element.style.backgroundPosition =
                      this.Reveal.getConfig().parallaxBackgroundPosition),
                    setTimeout(function () {
                      e.Reveal.getRevealElement().classList.add(
                        "has-parallax-background",
                      );
                    }, 1))
                  : ((this.element.style.backgroundImage = ""),
                    this.Reveal.getRevealElement().classList.remove(
                      "has-parallax-background",
                    ));
            },
          },
          {
            key: "createBackground",
            value: function (e, t) {
              var n = document.createElement("div");
              n.className =
                "slide-background " +
                e.className.replace(/present|past|future/, "");
              var i = document.createElement("div");
              return (
                (i.className = "slide-background-content"),
                n.appendChild(i),
                t.appendChild(n),
                (e.slideBackgroundElement = n),
                (e.slideBackgroundContentElement = i),
                this.sync(e),
                n
              );
            },
          },
          {
            key: "sync",
            value: function (e) {
              var t = e.slideBackgroundElement,
                n = e.slideBackgroundContentElement,
                i = {
                  background: e.getAttribute("data-background"),
                  backgroundSize: e.getAttribute("data-background-size"),
                  backgroundImage: e.getAttribute("data-background-image"),
                  backgroundVideo: e.getAttribute("data-background-video"),
                  backgroundIframe: e.getAttribute("data-background-iframe"),
                  backgroundColor: e.getAttribute("data-background-color"),
                  backgroundRepeat: e.getAttribute("data-background-repeat"),
                  backgroundPosition: e.getAttribute(
                    "data-background-position",
                  ),
                  backgroundTransition: e.getAttribute(
                    "data-background-transition",
                  ),
                  backgroundOpacity: e.getAttribute("data-background-opacity"),
                },
                r = e.hasAttribute("data-preload");
              e.classList.remove("has-dark-background"),
                e.classList.remove("has-light-background"),
                t.removeAttribute("data-loaded"),
                t.removeAttribute("data-background-hash"),
                t.removeAttribute("data-background-size"),
                t.removeAttribute("data-background-transition"),
                (t.style.backgroundColor = ""),
                (n.style.backgroundSize = ""),
                (n.style.backgroundRepeat = ""),
                (n.style.backgroundPosition = ""),
                (n.style.backgroundImage = ""),
                (n.style.opacity = ""),
                (n.innerHTML = ""),
                i.background &&
                  (/^(http|file|\/\/)/gi.test(i.background) ||
                  /\.(svg|png|jpg|jpeg|gif|bmp)([?#\s]|$)/gi.test(i.background)
                    ? e.setAttribute("data-background-image", i.background)
                    : (t.style.background = i.background)),
                (i.background ||
                  i.backgroundColor ||
                  i.backgroundImage ||
                  i.backgroundVideo ||
                  i.backgroundIframe) &&
                  t.setAttribute(
                    "data-background-hash",
                    i.background +
                      i.backgroundSize +
                      i.backgroundImage +
                      i.backgroundVideo +
                      i.backgroundIframe +
                      i.backgroundColor +
                      i.backgroundRepeat +
                      i.backgroundPosition +
                      i.backgroundTransition +
                      i.backgroundOpacity,
                  ),
                i.backgroundSize &&
                  t.setAttribute("data-background-size", i.backgroundSize),
                i.backgroundColor &&
                  (t.style.backgroundColor = i.backgroundColor),
                i.backgroundTransition &&
                  t.setAttribute(
                    "data-background-transition",
                    i.backgroundTransition,
                  ),
                r && t.setAttribute("data-preload", ""),
                i.backgroundSize && (n.style.backgroundSize = i.backgroundSize),
                i.backgroundRepeat &&
                  (n.style.backgroundRepeat = i.backgroundRepeat),
                i.backgroundPosition &&
                  (n.style.backgroundPosition = i.backgroundPosition),
                i.backgroundOpacity && (n.style.opacity = i.backgroundOpacity);
              var a,
                o = i.backgroundColor;
              if (!o || !Gh(o)) {
                var s = window.getComputedStyle(t);
                s && s.backgroundColor && (o = s.backgroundColor);
              }
              if (o) {
                var l = Gh(o);
                l &&
                  0 !== l.a &&
                  ("string" == typeof (a = o) && (a = Gh(a)),
                  (a ? (299 * a.r + 587 * a.g + 114 * a.b) / 1e3 : null) < 128
                    ? e.classList.add("has-dark-background")
                    : e.classList.add("has-light-background"));
              }
            },
          },
          {
            key: "update",
            value: function () {
              var e = this,
                t =
                  arguments.length > 0 &&
                  void 0 !== arguments[0] &&
                  arguments[0],
                n = this.Reveal.getCurrentSlide(),
                i = this.Reveal.getIndices(),
                r = null,
                a = this.Reveal.getConfig().rtl ? "future" : "past",
                o = this.Reveal.getConfig().rtl ? "past" : "future";
              if (
                (Array.from(this.element.childNodes).forEach(function (e, n) {
                  e.classList.remove("past", "present", "future"),
                    n < i.h
                      ? e.classList.add(a)
                      : n > i.h
                      ? e.classList.add(o)
                      : (e.classList.add("present"), (r = e)),
                    (t || n === i.h) &&
                      yh(e, ".slide-background").forEach(function (e, t) {
                        e.classList.remove("past", "present", "future"),
                          t < i.v
                            ? e.classList.add("past")
                            : t > i.v
                            ? e.classList.add("future")
                            : (e.classList.add("present"),
                              n === i.h && (r = e));
                      });
                }),
                this.previousBackground &&
                  this.Reveal.slideContent.stopEmbeddedContent(
                    this.previousBackground,
                    {
                      unloadIframes: !this.Reveal.slideContent.shouldPreload(
                        this.previousBackground,
                      ),
                    },
                  ),
                r)
              ) {
                this.Reveal.slideContent.startEmbeddedContent(r);
                var s = r.querySelector(".slide-background-content");
                if (s) {
                  var l = s.style.backgroundImage || "";
                  /\.gif/i.test(l) &&
                    ((s.style.backgroundImage = ""),
                    window.getComputedStyle(s).opacity,
                    (s.style.backgroundImage = l));
                }
                var c = this.previousBackground
                    ? this.previousBackground.getAttribute(
                        "data-background-hash",
                      )
                    : null,
                  u = r.getAttribute("data-background-hash");
                u &&
                  u === c &&
                  r !== this.previousBackground &&
                  this.element.classList.add("no-transition"),
                  (this.previousBackground = r);
              }
              n &&
                ["has-light-background", "has-dark-background"].forEach(
                  function (t) {
                    n.classList.contains(t)
                      ? e.Reveal.getRevealElement().classList.add(t)
                      : e.Reveal.getRevealElement().classList.remove(t);
                  },
                  this,
                ),
                setTimeout(function () {
                  e.element.classList.remove("no-transition");
                }, 1);
            },
          },
          {
            key: "updateParallax",
            value: function () {
              var e = this.Reveal.getIndices();
              if (this.Reveal.getConfig().parallaxBackgroundImage) {
                var t,
                  n,
                  i = this.Reveal.getHorizontalSlides(),
                  r = this.Reveal.getVerticalSlides(),
                  a = this.element.style.backgroundSize.split(" ");
                1 === a.length
                  ? (t = n = parseInt(a[0], 10))
                  : ((t = parseInt(a[0], 10)), (n = parseInt(a[1], 10)));
                var o,
                  s = this.element.offsetWidth,
                  l = i.length;
                o =
                  ("number" ==
                  typeof this.Reveal.getConfig().parallaxBackgroundHorizontal
                    ? this.Reveal.getConfig().parallaxBackgroundHorizontal
                    : l > 1
                    ? (t - s) / (l - 1)
                    : 0) *
                  e.h *
                  -1;
                var c,
                  u,
                  d = this.element.offsetHeight,
                  h = r.length;
                (c =
                  "number" ==
                  typeof this.Reveal.getConfig().parallaxBackgroundVertical
                    ? this.Reveal.getConfig().parallaxBackgroundVertical
                    : (n - d) / (h - 1)),
                  (u = h > 0 ? c * e.v : 0),
                  (this.element.style.backgroundPosition =
                    o + "px " + -u + "px");
              }
            },
          },
          {
            key: "destroy",
            value: function () {
              this.element.remove();
            },
          },
        ]),
        e
      );
    })(),
    Qh = A,
    Zh = Jt;
  Xt(
    {
      target: "Object",
      stat: !0,
      forced: r(function () {
        Zh(1);
      }),
    },
    {
      keys: function (e) {
        return Zh(Qh(e));
      },
    },
  );
  var ef = Yn.filter;
  Xt(
    { target: "Array", proto: !0, forced: !Jn("filter") },
    {
      filter: function (e) {
        return ef(this, e, arguments.length > 1 ? arguments[1] : void 0);
      },
    },
  );
  var tf = ".slides section",
    nf = ".slides>section",
    rf = ".slides>section.present>section",
    af =
      /registerPlugin|registerKeyboardShortcut|addKeyBinding|addEventListener/,
    of =
      /fade-(down|up|right|left|out|in-then-out|in-then-semi-out)|semi-fade-out|current-visible|shrink|grow/,
    sf = 0,
    lf = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      return (
        Ri(e, [
          {
            key: "run",
            value: function (e, t) {
              var n = this;
              this.reset();
              var i = this.Reveal.getSlides(),
                r = i.indexOf(t),
                a = i.indexOf(e);
              if (
                e.hasAttribute("data-auto-animate") &&
                t.hasAttribute("data-auto-animate") &&
                e.getAttribute("data-auto-animate-id") ===
                  t.getAttribute("data-auto-animate-id") &&
                !(r > a ? t : e).hasAttribute("data-auto-animate-restart")
              ) {
                this.autoAnimateStyleSheet = this.autoAnimateStyleSheet || Rh();
                var o = this.getAutoAnimateOptions(t);
                (e.dataset.autoAnimate = "pending"),
                  (t.dataset.autoAnimate = "pending"),
                  (o.slideDirection = r > a ? "forward" : "backward");
                var s = this.getAutoAnimatableElements(e, t).map(function (e) {
                  return n.autoAnimateElements(
                    e.from,
                    e.to,
                    e.options || {},
                    o,
                    sf++,
                  );
                });
                if (
                  "false" !== t.dataset.autoAnimateUnmatched &&
                  !0 === this.Reveal.getConfig().autoAnimateUnmatched
                ) {
                  var l = 0.8 * o.duration,
                    c = 0.2 * o.duration;
                  this.getUnmatchedAutoAnimateElements(t).forEach(function (e) {
                    var t = n.getAutoAnimateOptions(e, o),
                      i = "unmatched";
                    (t.duration === o.duration && t.delay === o.delay) ||
                      ((i = "unmatched-" + sf++),
                      s.push(
                        '[data-auto-animate="running"] [data-auto-animate-target="'
                          .concat(i, '"] { transition: opacity ')
                          .concat(t.duration, "s ease ")
                          .concat(t.delay, "s; }"),
                      )),
                      (e.dataset.autoAnimateTarget = i);
                  }, this),
                    s.push(
                      '[data-auto-animate="running"] [data-auto-animate-target="unmatched"] { transition: opacity '
                        .concat(l, "s ease ")
                        .concat(c, "s; }"),
                    );
                }
                (this.autoAnimateStyleSheet.innerHTML = s.join("")),
                  requestAnimationFrame(function () {
                    n.autoAnimateStyleSheet &&
                      (getComputedStyle(n.autoAnimateStyleSheet).fontWeight,
                      (t.dataset.autoAnimate = "running"));
                  }),
                  this.Reveal.dispatchEvent({
                    type: "autoanimate",
                    data: {
                      fromSlide: e,
                      toSlide: t,
                      sheet: this.autoAnimateStyleSheet,
                    },
                  });
              }
            },
          },
          {
            key: "reset",
            value: function () {
              yh(
                this.Reveal.getRevealElement(),
                '[data-auto-animate]:not([data-auto-animate=""])',
              ).forEach(function (e) {
                e.dataset.autoAnimate = "";
              }),
                yh(
                  this.Reveal.getRevealElement(),
                  "[data-auto-animate-target]",
                ).forEach(function (e) {
                  delete e.dataset.autoAnimateTarget;
                }),
                this.autoAnimateStyleSheet &&
                  this.autoAnimateStyleSheet.parentNode &&
                  (this.autoAnimateStyleSheet.parentNode.removeChild(
                    this.autoAnimateStyleSheet,
                  ),
                  (this.autoAnimateStyleSheet = null));
            },
          },
          {
            key: "autoAnimateElements",
            value: function (e, t, n, i, r) {
              (e.dataset.autoAnimateTarget = ""),
                (t.dataset.autoAnimateTarget = r);
              var a = this.getAutoAnimateOptions(t, i);
              void 0 !== n.delay && (a.delay = n.delay),
                void 0 !== n.duration && (a.duration = n.duration),
                void 0 !== n.easing && (a.easing = n.easing);
              var o = this.getAutoAnimatableProperties("from", e, n),
                s = this.getAutoAnimatableProperties("to", t, n);
              t.classList.contains("fragment") &&
                (delete s.styles.opacity,
                e.classList.contains("fragment") &&
                  (e.className.match(of) || [""])[0] ===
                    (t.className.match(of) || [""])[0] &&
                  "forward" === i.slideDirection &&
                  t.classList.add("visible", "disabled"));
              if (!1 !== n.translate || !1 !== n.scale) {
                var l = this.Reveal.getScale(),
                  c = {
                    x: (o.x - s.x) / l,
                    y: (o.y - s.y) / l,
                    scaleX: o.width / s.width,
                    scaleY: o.height / s.height,
                  };
                (c.x = Math.round(1e3 * c.x) / 1e3),
                  (c.y = Math.round(1e3 * c.y) / 1e3),
                  (c.scaleX = Math.round(1e3 * c.scaleX) / 1e3),
                  (c.scaleX = Math.round(1e3 * c.scaleX) / 1e3);
                var u = !1 !== n.translate && (0 !== c.x || 0 !== c.y),
                  d = !1 !== n.scale && (0 !== c.scaleX || 0 !== c.scaleY);
                if (u || d) {
                  var h = [];
                  u &&
                    h.push("translate(".concat(c.x, "px, ").concat(c.y, "px)")),
                    d &&
                      h.push(
                        "scale(".concat(c.scaleX, ", ").concat(c.scaleY, ")"),
                      ),
                    (o.styles.transform = h.join(" ")),
                    (o.styles["transform-origin"] = "top left"),
                    (s.styles.transform = "none");
                }
              }
              for (var f in s.styles) {
                var v = s.styles[f],
                  p = o.styles[f];
                v === p
                  ? delete s.styles[f]
                  : (!0 === v.explicitValue && (s.styles[f] = v.value),
                    !0 === p.explicitValue && (o.styles[f] = p.value));
              }
              var g = "",
                m = Object.keys(s.styles);
              m.length > 0 &&
                ((o.styles.transition = "none"),
                (s.styles.transition = "all "
                  .concat(a.duration, "s ")
                  .concat(a.easing, " ")
                  .concat(a.delay, "s")),
                (s.styles["transition-property"] = m.join(", ")),
                (s.styles["will-change"] = m.join(", ")),
                (g =
                  '[data-auto-animate-target="' +
                  r +
                  '"] {' +
                  Object.keys(o.styles)
                    .map(function (e) {
                      return e + ": " + o.styles[e] + " !important;";
                    })
                    .join("") +
                  '}[data-auto-animate="running"] [data-auto-animate-target="' +
                  r +
                  '"] {' +
                  Object.keys(s.styles)
                    .map(function (e) {
                      return e + ": " + s.styles[e] + " !important;";
                    })
                    .join("") +
                  "}"));
              return g;
            },
          },
          {
            key: "getAutoAnimateOptions",
            value: function (e, t) {
              var n = {
                easing: this.Reveal.getConfig().autoAnimateEasing,
                duration: this.Reveal.getConfig().autoAnimateDuration,
                delay: 0,
              };
              if (((n = mh(n, t)), e.parentNode)) {
                var i = kh(e.parentNode, "[data-auto-animate-target]");
                i && (n = this.getAutoAnimateOptions(i, n));
              }
              return (
                e.dataset.autoAnimateEasing &&
                  (n.easing = e.dataset.autoAnimateEasing),
                e.dataset.autoAnimateDuration &&
                  (n.duration = parseFloat(e.dataset.autoAnimateDuration)),
                e.dataset.autoAnimateDelay &&
                  (n.delay = parseFloat(e.dataset.autoAnimateDelay)),
                n
              );
            },
          },
          {
            key: "getAutoAnimatableProperties",
            value: function (e, t, n) {
              var i = this.Reveal.getConfig(),
                r = { styles: [] };
              if (!1 !== n.translate || !1 !== n.scale) {
                var a;
                if ("function" == typeof n.measure) a = n.measure(t);
                else if (i.center) a = t.getBoundingClientRect();
                else {
                  var o = this.Reveal.getScale();
                  a = {
                    x: t.offsetLeft * o,
                    y: t.offsetTop * o,
                    width: t.offsetWidth * o,
                    height: t.offsetHeight * o,
                  };
                }
                (r.x = a.x),
                  (r.y = a.y),
                  (r.width = a.width),
                  (r.height = a.height);
              }
              var s = getComputedStyle(t);
              return (
                (n.styles || i.autoAnimateStyles).forEach(function (t) {
                  var n;
                  "string" == typeof t && (t = { property: t }),
                    "" !==
                      (n =
                        void 0 !== t.from && "from" === e
                          ? { value: t.from, explicitValue: !0 }
                          : void 0 !== t.to && "to" === e
                          ? { value: t.to, explicitValue: !0 }
                          : s[t.property]) && (r.styles[t.property] = n);
                }),
                r
              );
            },
          },
          {
            key: "getAutoAnimatableElements",
            value: function (e, t) {
              var n = (
                  "function" ==
                  typeof this.Reveal.getConfig().autoAnimateMatcher
                    ? this.Reveal.getConfig().autoAnimateMatcher
                    : this.getAutoAnimatePairs
                ).call(this, e, t),
                i = [];
              return n.filter(function (e, t) {
                if (-1 === i.indexOf(e.to)) return i.push(e.to), !0;
              });
            },
          },
          {
            key: "getAutoAnimatePairs",
            value: function (e, t) {
              var n = this,
                i = [],
                r = "h1, h2, h3, h4, h5, h6, p, li";
              return (
                this.findAutoAnimateMatches(i, e, t, "[data-id]", function (e) {
                  return e.nodeName + ":::" + e.getAttribute("data-id");
                }),
                this.findAutoAnimateMatches(i, e, t, r, function (e) {
                  return e.nodeName + ":::" + e.innerText;
                }),
                this.findAutoAnimateMatches(
                  i,
                  e,
                  t,
                  "img, video, iframe",
                  function (e) {
                    return (
                      e.nodeName +
                      ":::" +
                      (e.getAttribute("src") || e.getAttribute("data-src"))
                    );
                  },
                ),
                this.findAutoAnimateMatches(i, e, t, "pre", function (e) {
                  return e.nodeName + ":::" + e.innerText;
                }),
                i.forEach(function (e) {
                  Sh(e.from, r)
                    ? (e.options = { scale: !1 })
                    : Sh(e.from, "pre") &&
                      ((e.options = { scale: !1, styles: ["width", "height"] }),
                      n.findAutoAnimateMatches(
                        i,
                        e.from,
                        e.to,
                        ".hljs .hljs-ln-code",
                        function (e) {
                          return e.textContent;
                        },
                        {
                          scale: !1,
                          styles: [],
                          measure: n.getLocalBoundingBox.bind(n),
                        },
                      ),
                      n.findAutoAnimateMatches(
                        i,
                        e.from,
                        e.to,
                        ".hljs .hljs-ln-line[data-line-number]",
                        function (e) {
                          return e.getAttribute("data-line-number");
                        },
                        {
                          scale: !1,
                          styles: ["width"],
                          measure: n.getLocalBoundingBox.bind(n),
                        },
                      ));
                }, this),
                i
              );
            },
          },
          {
            key: "getLocalBoundingBox",
            value: function (e) {
              var t = this.Reveal.getScale();
              return {
                x: Math.round(e.offsetLeft * t * 100) / 100,
                y: Math.round(e.offsetTop * t * 100) / 100,
                width: Math.round(e.offsetWidth * t * 100) / 100,
                height: Math.round(e.offsetHeight * t * 100) / 100,
              };
            },
          },
          {
            key: "findAutoAnimateMatches",
            value: function (e, t, n, i, r, a) {
              var o = {},
                s = {};
              [].slice.call(t.querySelectorAll(i)).forEach(function (e, t) {
                var n = r(e);
                "string" == typeof n &&
                  n.length &&
                  ((o[n] = o[n] || []), o[n].push(e));
              }),
                [].slice.call(n.querySelectorAll(i)).forEach(function (t, n) {
                  var i,
                    l = r(t);
                  if (((s[l] = s[l] || []), s[l].push(t), o[l])) {
                    var c = s[l].length - 1,
                      u = o[l].length - 1;
                    o[l][c]
                      ? ((i = o[l][c]), (o[l][c] = null))
                      : o[l][u] && ((i = o[l][u]), (o[l][u] = null));
                  }
                  i && e.push({ from: i, to: t, options: a });
                });
            },
          },
          {
            key: "getUnmatchedAutoAnimateElements",
            value: function (e) {
              var t = this;
              return [].slice.call(e.children).reduce(function (e, n) {
                var i = n.querySelector("[data-auto-animate-target]");
                return (
                  n.hasAttribute("data-auto-animate-target") || i || e.push(n),
                  n.querySelector("[data-auto-animate-target]") &&
                    (e = e.concat(t.getUnmatchedAutoAnimateElements(n))),
                  e
                );
              }, []);
            },
          },
        ]),
        e
      );
    })(),
    cf = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      return (
        Ri(e, [
          {
            key: "configure",
            value: function (e, t) {
              !1 === e.fragments
                ? this.disable()
                : !1 === t.fragments && this.enable();
            },
          },
          {
            key: "disable",
            value: function () {
              yh(this.Reveal.getSlidesElement(), ".fragment").forEach(
                function (e) {
                  e.classList.add("visible"),
                    e.classList.remove("current-fragment");
                },
              );
            },
          },
          {
            key: "enable",
            value: function () {
              yh(this.Reveal.getSlidesElement(), ".fragment").forEach(
                function (e) {
                  e.classList.remove("visible"),
                    e.classList.remove("current-fragment");
                },
              );
            },
          },
          {
            key: "availableRoutes",
            value: function () {
              var e = this.Reveal.getCurrentSlide();
              if (e && this.Reveal.getConfig().fragments) {
                var t = e.querySelectorAll(".fragment:not(.disabled)"),
                  n = e.querySelectorAll(
                    ".fragment:not(.disabled):not(.visible)",
                  );
                return { prev: t.length - n.length > 0, next: !!n.length };
              }
              return { prev: !1, next: !1 };
            },
          },
          {
            key: "sort",
            value: function (e) {
              var t =
                arguments.length > 1 && void 0 !== arguments[1] && arguments[1];
              e = Array.from(e);
              var n = [],
                i = [],
                r = [];
              e.forEach(function (e) {
                if (e.hasAttribute("data-fragment-index")) {
                  var t = parseInt(e.getAttribute("data-fragment-index"), 10);
                  n[t] || (n[t] = []), n[t].push(e);
                } else i.push([e]);
              }),
                (n = n.concat(i));
              var a = 0;
              return (
                n.forEach(function (e) {
                  e.forEach(function (e) {
                    r.push(e), e.setAttribute("data-fragment-index", a);
                  }),
                    a++;
                }),
                !0 === t ? n : r
              );
            },
          },
          {
            key: "sortAll",
            value: function () {
              var e = this;
              this.Reveal.getHorizontalSlides().forEach(function (t) {
                var n = yh(t, "section");
                n.forEach(function (t, n) {
                  e.sort(t.querySelectorAll(".fragment"));
                }, e),
                  0 === n.length && e.sort(t.querySelectorAll(".fragment"));
              });
            },
          },
          {
            key: "update",
            value: function (e, t) {
              var n = this,
                i = { shown: [], hidden: [] },
                r = this.Reveal.getCurrentSlide();
              if (
                r &&
                this.Reveal.getConfig().fragments &&
                (t = t || this.sort(r.querySelectorAll(".fragment"))).length
              ) {
                var a = 0;
                if ("number" != typeof e) {
                  var o = this.sort(
                    r.querySelectorAll(".fragment.visible"),
                  ).pop();
                  o &&
                    (e = parseInt(
                      o.getAttribute("data-fragment-index") || 0,
                      10,
                    ));
                }
                Array.from(t).forEach(function (t, r) {
                  if (
                    (t.hasAttribute("data-fragment-index") &&
                      (r = parseInt(t.getAttribute("data-fragment-index"), 10)),
                    (a = Math.max(a, r)),
                    r <= e)
                  ) {
                    var o = t.classList.contains("visible");
                    t.classList.add("visible"),
                      t.classList.remove("current-fragment"),
                      r === e &&
                        (n.Reveal.announceStatus(n.Reveal.getStatusText(t)),
                        t.classList.add("current-fragment"),
                        n.Reveal.slideContent.startEmbeddedContent(t)),
                      o ||
                        (i.shown.push(t),
                        n.Reveal.dispatchEvent({
                          target: t,
                          type: "visible",
                          bubbles: !1,
                        }));
                  } else {
                    var s = t.classList.contains("visible");
                    t.classList.remove("visible"),
                      t.classList.remove("current-fragment"),
                      s &&
                        (n.Reveal.slideContent.stopEmbeddedContent(t),
                        i.hidden.push(t),
                        n.Reveal.dispatchEvent({
                          target: t,
                          type: "hidden",
                          bubbles: !1,
                        }));
                  }
                }),
                  (e = "number" == typeof e ? e : -1),
                  (e = Math.max(Math.min(e, a), -1)),
                  r.setAttribute("data-fragment", e);
              }
              return i;
            },
          },
          {
            key: "sync",
            value: function () {
              var e =
                arguments.length > 0 && void 0 !== arguments[0]
                  ? arguments[0]
                  : this.Reveal.getCurrentSlide();
              return this.sort(e.querySelectorAll(".fragment"));
            },
          },
          {
            key: "goto",
            value: function (e) {
              var t =
                  arguments.length > 1 && void 0 !== arguments[1]
                    ? arguments[1]
                    : 0,
                n = this.Reveal.getCurrentSlide();
              if (n && this.Reveal.getConfig().fragments) {
                var i = this.sort(
                  n.querySelectorAll(".fragment:not(.disabled)"),
                );
                if (i.length) {
                  if ("number" != typeof e) {
                    var r = this.sort(
                      n.querySelectorAll(".fragment:not(.disabled).visible"),
                    ).pop();
                    e = r
                      ? parseInt(r.getAttribute("data-fragment-index") || 0, 10)
                      : -1;
                  }
                  e += t;
                  var a = this.update(e, i);
                  return (
                    a.hidden.length &&
                      this.Reveal.dispatchEvent({
                        type: "fragmenthidden",
                        data: { fragment: a.hidden[0], fragments: a.hidden },
                      }),
                    a.shown.length &&
                      this.Reveal.dispatchEvent({
                        type: "fragmentshown",
                        data: { fragment: a.shown[0], fragments: a.shown },
                      }),
                    this.Reveal.controls.update(),
                    this.Reveal.progress.update(),
                    this.Reveal.getConfig().fragmentInURL &&
                      this.Reveal.location.writeURL(),
                    !(!a.shown.length && !a.hidden.length)
                  );
                }
              }
              return !1;
            },
          },
          {
            key: "next",
            value: function () {
              return this.goto(null, 1);
            },
          },
          {
            key: "prev",
            value: function () {
              return this.goto(null, -1);
            },
          },
        ]),
        e
      );
    })(),
    uf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.active = !1),
          (this.onSlideClicked = this.onSlideClicked.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "activate",
            value: function () {
              var e = this;
              if (this.Reveal.getConfig().overview && !this.isActive()) {
                (this.active = !0),
                  this.Reveal.getRevealElement().classList.add("overview"),
                  this.Reveal.cancelAutoSlide(),
                  this.Reveal.getSlidesElement().appendChild(
                    this.Reveal.getBackgroundsElement(),
                  ),
                  yh(this.Reveal.getRevealElement(), tf).forEach(function (t) {
                    t.classList.contains("stack") ||
                      t.addEventListener("click", e.onSlideClicked, !0);
                  });
                var t = this.Reveal.getComputedSlideSize();
                (this.overviewSlideWidth = t.width + 70),
                  (this.overviewSlideHeight = t.height + 70),
                  this.Reveal.getConfig().rtl &&
                    (this.overviewSlideWidth = -this.overviewSlideWidth),
                  this.Reveal.updateSlidesVisibility(),
                  this.layout(),
                  this.update(),
                  this.Reveal.layout();
                var n = this.Reveal.getIndices();
                this.Reveal.dispatchEvent({
                  type: "overviewshown",
                  data: {
                    indexh: n.h,
                    indexv: n.v,
                    currentSlide: this.Reveal.getCurrentSlide(),
                  },
                });
              }
            },
          },
          {
            key: "layout",
            value: function () {
              var e = this;
              this.Reveal.getHorizontalSlides().forEach(function (t, n) {
                t.setAttribute("data-index-h", n),
                  Eh(
                    t,
                    "translate3d(" + n * e.overviewSlideWidth + "px, 0, 0)",
                  ),
                  t.classList.contains("stack") &&
                    yh(t, "section").forEach(function (t, i) {
                      t.setAttribute("data-index-h", n),
                        t.setAttribute("data-index-v", i),
                        Eh(
                          t,
                          "translate3d(0, " +
                            i * e.overviewSlideHeight +
                            "px, 0)",
                        );
                    });
              }),
                Array.from(
                  this.Reveal.getBackgroundsElement().childNodes,
                ).forEach(function (t, n) {
                  Eh(
                    t,
                    "translate3d(" + n * e.overviewSlideWidth + "px, 0, 0)",
                  ),
                    yh(t, ".slide-background").forEach(function (t, n) {
                      Eh(
                        t,
                        "translate3d(0, " +
                          n * e.overviewSlideHeight +
                          "px, 0)",
                      );
                    });
                });
            },
          },
          {
            key: "update",
            value: function () {
              var e = Math.min(window.innerWidth, window.innerHeight),
                t = Math.max(e / 5, 150) / e,
                n = this.Reveal.getIndices();
              this.Reveal.transformSlides({
                overview: [
                  "scale(" + t + ")",
                  "translateX(" + -n.h * this.overviewSlideWidth + "px)",
                  "translateY(" + -n.v * this.overviewSlideHeight + "px)",
                ].join(" "),
              });
            },
          },
          {
            key: "deactivate",
            value: function () {
              var e = this;
              if (this.Reveal.getConfig().overview) {
                (this.active = !1),
                  this.Reveal.getRevealElement().classList.remove("overview"),
                  this.Reveal.getRevealElement().classList.add(
                    "overview-deactivating",
                  ),
                  setTimeout(function () {
                    e.Reveal.getRevealElement().classList.remove(
                      "overview-deactivating",
                    );
                  }, 1),
                  this.Reveal.getRevealElement().appendChild(
                    this.Reveal.getBackgroundsElement(),
                  ),
                  yh(this.Reveal.getRevealElement(), tf).forEach(function (t) {
                    Eh(t, ""),
                      t.removeEventListener("click", e.onSlideClicked, !0);
                  }),
                  yh(
                    this.Reveal.getBackgroundsElement(),
                    ".slide-background",
                  ).forEach(function (e) {
                    Eh(e, "");
                  }),
                  this.Reveal.transformSlides({ overview: "" });
                var t = this.Reveal.getIndices();
                this.Reveal.slide(t.h, t.v),
                  this.Reveal.layout(),
                  this.Reveal.cueAutoSlide(),
                  this.Reveal.dispatchEvent({
                    type: "overviewhidden",
                    data: {
                      indexh: t.h,
                      indexv: t.v,
                      currentSlide: this.Reveal.getCurrentSlide(),
                    },
                  });
              }
            },
          },
          {
            key: "toggle",
            value: function (e) {
              "boolean" == typeof e
                ? e
                  ? this.activate()
                  : this.deactivate()
                : this.isActive()
                ? this.deactivate()
                : this.activate();
            },
          },
          {
            key: "isActive",
            value: function () {
              return this.active;
            },
          },
          {
            key: "onSlideClicked",
            value: function (e) {
              if (this.isActive()) {
                e.preventDefault();
                for (var t = e.target; t && !t.nodeName.match(/section/gi); )
                  t = t.parentNode;
                if (
                  t &&
                  !t.classList.contains("disabled") &&
                  (this.deactivate(), t.nodeName.match(/section/gi))
                ) {
                  var n = parseInt(t.getAttribute("data-index-h"), 10),
                    i = parseInt(t.getAttribute("data-index-v"), 10);
                  this.Reveal.slide(n, i);
                }
              }
            },
          },
        ]),
        e
      );
    })(),
    df = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.shortcuts = {}),
          (this.bindings = {}),
          (this.onDocumentKeyDown = this.onDocumentKeyDown.bind(this)),
          (this.onDocumentKeyPress = this.onDocumentKeyPress.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "configure",
            value: function (e, t) {
              "linear" === e.navigationMode
                ? ((this.shortcuts[
                    "&#8594;  ,  &#8595;  ,  SPACE  ,  N  ,  L  ,  J"
                  ] = "Next slide"),
                  (this.shortcuts["&#8592;  ,  &#8593;  ,  P  ,  H  ,  K"] =
                    "Previous slide"))
                : ((this.shortcuts["N  ,  SPACE"] = "Next slide"),
                  (this.shortcuts["P  ,  Shift SPACE"] = "Previous slide"),
                  (this.shortcuts["&#8592;  ,  H"] = "Navigate left"),
                  (this.shortcuts["&#8594;  ,  L"] = "Navigate right"),
                  (this.shortcuts["&#8593;  ,  K"] = "Navigate up"),
                  (this.shortcuts["&#8595;  ,  J"] = "Navigate down")),
                (this.shortcuts["Alt + &#8592;/&#8593/&#8594;/&#8595;"] =
                  "Navigate without fragments"),
                (this.shortcuts["Shift + &#8592;/&#8593/&#8594;/&#8595;"] =
                  "Jump to first/last slide"),
                (this.shortcuts["B  ,  ."] = "Pause"),
                (this.shortcuts.F = "Fullscreen"),
                (this.shortcuts["ESC, O"] = "Slide overview");
            },
          },
          {
            key: "bind",
            value: function () {
              document.addEventListener("keydown", this.onDocumentKeyDown, !1),
                document.addEventListener(
                  "keypress",
                  this.onDocumentKeyPress,
                  !1,
                );
            },
          },
          {
            key: "unbind",
            value: function () {
              document.removeEventListener(
                "keydown",
                this.onDocumentKeyDown,
                !1,
              ),
                document.removeEventListener(
                  "keypress",
                  this.onDocumentKeyPress,
                  !1,
                );
            },
          },
          {
            key: "addKeyBinding",
            value: function (e, t) {
              "object" === Ei(e) && e.keyCode
                ? (this.bindings[e.keyCode] = {
                    callback: t,
                    key: e.key,
                    description: e.description,
                  })
                : (this.bindings[e] = {
                    callback: t,
                    key: null,
                    description: null,
                  });
            },
          },
          {
            key: "removeKeyBinding",
            value: function (e) {
              delete this.bindings[e];
            },
          },
          {
            key: "triggerKey",
            value: function (e) {
              this.onDocumentKeyDown({ keyCode: e });
            },
          },
          {
            key: "registerKeyboardShortcut",
            value: function (e, t) {
              this.shortcuts[e] = t;
            },
          },
          {
            key: "getShortcuts",
            value: function () {
              return this.shortcuts;
            },
          },
          {
            key: "getBindings",
            value: function () {
              return this.bindings;
            },
          },
          {
            key: "onDocumentKeyPress",
            value: function (e) {
              e.shiftKey && 63 === e.charCode && this.Reveal.toggleHelp();
            },
          },
          {
            key: "onDocumentKeyDown",
            value: function (e) {
              var t = this.Reveal.getConfig();
              if (
                "function" == typeof t.keyboardCondition &&
                !1 === t.keyboardCondition(e)
              )
                return !0;
              if ("focused" === t.keyboardCondition && !this.Reveal.isFocused())
                return !0;
              var n = e.keyCode,
                i = !this.Reveal.isAutoSliding();
              this.Reveal.onUserInput(e);
              var r =
                  document.activeElement &&
                  !0 === document.activeElement.isContentEditable,
                a =
                  document.activeElement &&
                  document.activeElement.tagName &&
                  /input|textarea/i.test(document.activeElement.tagName),
                o =
                  document.activeElement &&
                  document.activeElement.className &&
                  /speaker-notes/i.test(document.activeElement.className),
                s =
                  !(
                    (-1 !== [32, 37, 38, 39, 40, 78, 80].indexOf(e.keyCode) &&
                      e.shiftKey) ||
                    e.altKey
                  ) &&
                  (e.shiftKey || e.altKey || e.ctrlKey || e.metaKey);
              if (!(r || a || o || s)) {
                var l,
                  c = [66, 86, 190, 191];
                if ("object" === Ei(t.keyboard))
                  for (l in t.keyboard)
                    "togglePause" === t.keyboard[l] && c.push(parseInt(l, 10));
                if (this.Reveal.isPaused() && -1 === c.indexOf(n)) return !1;
                var u,
                  d,
                  h =
                    "linear" === t.navigationMode ||
                    !this.Reveal.hasHorizontalSlides() ||
                    !this.Reveal.hasVerticalSlides(),
                  f = !1;
                if ("object" === Ei(t.keyboard))
                  for (l in t.keyboard)
                    if (parseInt(l, 10) === n) {
                      var v = t.keyboard[l];
                      "function" == typeof v
                        ? v.apply(null, [e])
                        : "string" == typeof v &&
                          "function" == typeof this.Reveal[v] &&
                          this.Reveal[v].call(),
                        (f = !0);
                    }
                if (!1 === f)
                  for (l in this.bindings)
                    if (parseInt(l, 10) === n) {
                      var p = this.bindings[l].callback;
                      "function" == typeof p
                        ? p.apply(null, [e])
                        : "string" == typeof p &&
                          "function" == typeof this.Reveal[p] &&
                          this.Reveal[p].call(),
                        (f = !0);
                    }
                !1 === f &&
                  ((f = !0),
                  80 === n || 33 === n
                    ? this.Reveal.prev({ skipFragments: e.altKey })
                    : 78 === n || 34 === n
                    ? this.Reveal.next({ skipFragments: e.altKey })
                    : 72 === n || 37 === n
                    ? e.shiftKey
                      ? this.Reveal.slide(0)
                      : !this.Reveal.overview.isActive() && h
                      ? this.Reveal.prev({ skipFragments: e.altKey })
                      : this.Reveal.left({ skipFragments: e.altKey })
                    : 76 === n || 39 === n
                    ? e.shiftKey
                      ? this.Reveal.slide(
                          this.Reveal.getHorizontalSlides().length - 1,
                        )
                      : !this.Reveal.overview.isActive() && h
                      ? this.Reveal.next({ skipFragments: e.altKey })
                      : this.Reveal.right({ skipFragments: e.altKey })
                    : 75 === n || 38 === n
                    ? e.shiftKey
                      ? this.Reveal.slide(void 0, 0)
                      : !this.Reveal.overview.isActive() && h
                      ? this.Reveal.prev({ skipFragments: e.altKey })
                      : this.Reveal.up({ skipFragments: e.altKey })
                    : 74 === n || 40 === n
                    ? e.shiftKey
                      ? this.Reveal.slide(void 0, Number.MAX_VALUE)
                      : !this.Reveal.overview.isActive() && h
                      ? this.Reveal.next({ skipFragments: e.altKey })
                      : this.Reveal.down({ skipFragments: e.altKey })
                    : 36 === n
                    ? this.Reveal.slide(0)
                    : 35 === n
                    ? this.Reveal.slide(
                        this.Reveal.getHorizontalSlides().length - 1,
                      )
                    : 32 === n
                    ? (this.Reveal.overview.isActive() &&
                        this.Reveal.overview.deactivate(),
                      e.shiftKey
                        ? this.Reveal.prev({ skipFragments: e.altKey })
                        : this.Reveal.next({ skipFragments: e.altKey }))
                    : 58 === n ||
                      59 === n ||
                      66 === n ||
                      86 === n ||
                      190 === n ||
                      191 === n
                    ? this.Reveal.togglePause()
                    : 70 === n
                    ? ((u = t.embedded
                        ? this.Reveal.getViewportElement()
                        : document.documentElement),
                      (d =
                        (u = u || document.documentElement).requestFullscreen ||
                        u.webkitRequestFullscreen ||
                        u.webkitRequestFullScreen ||
                        u.mozRequestFullScreen ||
                        u.msRequestFullscreen) && d.apply(u))
                    : 65 === n
                    ? t.autoSlideStoppable && this.Reveal.toggleAutoSlide(i)
                    : (f = !1)),
                  f
                    ? e.preventDefault && e.preventDefault()
                    : (27 !== n && 79 !== n) ||
                      (!1 === this.Reveal.closeOverlay() &&
                        this.Reveal.overview.toggle(),
                      e.preventDefault && e.preventDefault()),
                  this.Reveal.cueAutoSlide();
              }
            },
          },
        ]),
        e
      );
    })(),
    hf = (function () {
      function e(t) {
        ki(this, e),
          xi(this, "MAX_REPLACE_STATE_FREQUENCY", 1e3),
          (this.Reveal = t),
          (this.writeURLTimeout = 0),
          (this.replaceStateTimestamp = 0),
          (this.onWindowHashChange = this.onWindowHashChange.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "bind",
            value: function () {
              window.addEventListener(
                "hashchange",
                this.onWindowHashChange,
                !1,
              );
            },
          },
          {
            key: "unbind",
            value: function () {
              window.removeEventListener(
                "hashchange",
                this.onWindowHashChange,
                !1,
              );
            },
          },
          {
            key: "getIndicesFromHash",
            value: function () {
              var e,
                t,
                n =
                  arguments.length > 0 && void 0 !== arguments[0]
                    ? arguments[0]
                    : window.location.hash,
                i = n.replace(/^#\/?/, ""),
                r = i.split("/");
              if (/^[0-9]*$/.test(r[0]) || !i.length) {
                var a,
                  o = this.Reveal.getConfig(),
                  s = o.hashOneBasedIndex ? 1 : 0,
                  l = parseInt(r[0], 10) - s || 0,
                  c = parseInt(r[1], 10) - s || 0;
                return (
                  o.fragmentInURL &&
                    ((a = parseInt(r[2], 10)), isNaN(a) && (a = void 0)),
                  { h: l, v: c, f: a }
                );
              }
              /\/[-\d]+$/g.test(i) &&
                ((t = parseInt(i.split("/").pop(), 10)),
                (t = isNaN(t) ? void 0 : t),
                (i = i.split("/").shift()));
              try {
                e = document.getElementById(decodeURIComponent(i));
              } catch (e) {}
              return e
                ? wi(wi({}, this.Reveal.getIndices(e)), {}, { f: t })
                : null;
            },
          },
          {
            key: "readURL",
            value: function () {
              var e = this.Reveal.getIndices(),
                t = this.getIndicesFromHash();
              t
                ? (t.h === e.h && t.v === e.v && void 0 === t.f) ||
                  this.Reveal.slide(t.h, t.v, t.f)
                : this.Reveal.slide(e.h || 0, e.v || 0);
            },
          },
          {
            key: "writeURL",
            value: function (e) {
              var t = this.Reveal.getConfig(),
                n = this.Reveal.getCurrentSlide();
              if ((clearTimeout(this.writeURLTimeout), "number" == typeof e))
                this.writeURLTimeout = setTimeout(this.writeURL, e);
              else if (n) {
                var i = this.getHash();
                t.history
                  ? (window.location.hash = i)
                  : t.hash &&
                    ("/" === i
                      ? this.debouncedReplaceState(
                          window.location.pathname + window.location.search,
                        )
                      : this.debouncedReplaceState("#" + i));
              }
            },
          },
          {
            key: "replaceState",
            value: function (e) {
              window.history.replaceState(null, null, e),
                (this.replaceStateTimestamp = Date.now());
            },
          },
          {
            key: "debouncedReplaceState",
            value: function (e) {
              var t = this;
              clearTimeout(this.replaceStateTimeout),
                Date.now() - this.replaceStateTimestamp >
                this.MAX_REPLACE_STATE_FREQUENCY
                  ? this.replaceState(e)
                  : (this.replaceStateTimeout = setTimeout(function () {
                      return t.replaceState(e);
                    }, this.MAX_REPLACE_STATE_FREQUENCY));
            },
          },
          {
            key: "getHash",
            value: function (e) {
              var t = "/",
                n = e || this.Reveal.getCurrentSlide(),
                i = n ? n.getAttribute("id") : null;
              i && (i = encodeURIComponent(i));
              var r = this.Reveal.getIndices(e);
              if (
                (this.Reveal.getConfig().fragmentInURL || (r.f = void 0),
                "string" == typeof i && i.length)
              )
                (t = "/" + i), r.f >= 0 && (t += "/" + r.f);
              else {
                var a = this.Reveal.getConfig().hashOneBasedIndex ? 1 : 0;
                (r.h > 0 || r.v > 0 || r.f >= 0) && (t += r.h + a),
                  (r.v > 0 || r.f >= 0) && (t += "/" + (r.v + a)),
                  r.f >= 0 && (t += "/" + r.f);
              }
              return t;
            },
          },
          {
            key: "onWindowHashChange",
            value: function (e) {
              this.readURL();
            },
          },
        ]),
        e
      );
    })(),
    ff = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.onNavigateLeftClicked = this.onNavigateLeftClicked.bind(this)),
          (this.onNavigateRightClicked =
            this.onNavigateRightClicked.bind(this)),
          (this.onNavigateUpClicked = this.onNavigateUpClicked.bind(this)),
          (this.onNavigateDownClicked = this.onNavigateDownClicked.bind(this)),
          (this.onNavigatePrevClicked = this.onNavigatePrevClicked.bind(this)),
          (this.onNavigateNextClicked = this.onNavigateNextClicked.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "render",
            value: function () {
              var e = this.Reveal.getConfig().rtl,
                t = this.Reveal.getRevealElement();
              (this.element = document.createElement("aside")),
                (this.element.className = "controls"),
                (this.element.innerHTML =
                  '<button class="navigate-left" aria-label="'
                    .concat(
                      e ? "next slide" : "previous slide",
                      '"><div class="controls-arrow"></div></button>\n\t\t\t<button class="navigate-right" aria-label="',
                    )
                    .concat(
                      e ? "previous slide" : "next slide",
                      '"><div class="controls-arrow"></div></button>\n\t\t\t<button class="navigate-up" aria-label="above slide"><div class="controls-arrow"></div></button>\n\t\t\t<button class="navigate-down" aria-label="below slide"><div class="controls-arrow"></div></button>',
                    )),
                this.Reveal.getRevealElement().appendChild(this.element),
                (this.controlsLeft = yh(t, ".navigate-left")),
                (this.controlsRight = yh(t, ".navigate-right")),
                (this.controlsUp = yh(t, ".navigate-up")),
                (this.controlsDown = yh(t, ".navigate-down")),
                (this.controlsPrev = yh(t, ".navigate-prev")),
                (this.controlsNext = yh(t, ".navigate-next")),
                (this.controlsRightArrow =
                  this.element.querySelector(".navigate-right")),
                (this.controlsLeftArrow =
                  this.element.querySelector(".navigate-left")),
                (this.controlsDownArrow =
                  this.element.querySelector(".navigate-down"));
            },
          },
          {
            key: "configure",
            value: function (e, t) {
              (this.element.style.display = e.controls ? "block" : "none"),
                this.element.setAttribute(
                  "data-controls-layout",
                  e.controlsLayout,
                ),
                this.element.setAttribute(
                  "data-controls-back-arrows",
                  e.controlsBackArrows,
                );
            },
          },
          {
            key: "bind",
            value: function () {
              var e = this,
                t = ["touchstart", "click"];
              Ih && (t = ["touchstart"]),
                t.forEach(function (t) {
                  e.controlsLeft.forEach(function (n) {
                    return n.addEventListener(t, e.onNavigateLeftClicked, !1);
                  }),
                    e.controlsRight.forEach(function (n) {
                      return n.addEventListener(
                        t,
                        e.onNavigateRightClicked,
                        !1,
                      );
                    }),
                    e.controlsUp.forEach(function (n) {
                      return n.addEventListener(t, e.onNavigateUpClicked, !1);
                    }),
                    e.controlsDown.forEach(function (n) {
                      return n.addEventListener(t, e.onNavigateDownClicked, !1);
                    }),
                    e.controlsPrev.forEach(function (n) {
                      return n.addEventListener(t, e.onNavigatePrevClicked, !1);
                    }),
                    e.controlsNext.forEach(function (n) {
                      return n.addEventListener(t, e.onNavigateNextClicked, !1);
                    });
                });
            },
          },
          {
            key: "unbind",
            value: function () {
              var e = this;
              ["touchstart", "click"].forEach(function (t) {
                e.controlsLeft.forEach(function (n) {
                  return n.removeEventListener(t, e.onNavigateLeftClicked, !1);
                }),
                  e.controlsRight.forEach(function (n) {
                    return n.removeEventListener(
                      t,
                      e.onNavigateRightClicked,
                      !1,
                    );
                  }),
                  e.controlsUp.forEach(function (n) {
                    return n.removeEventListener(t, e.onNavigateUpClicked, !1);
                  }),
                  e.controlsDown.forEach(function (n) {
                    return n.removeEventListener(
                      t,
                      e.onNavigateDownClicked,
                      !1,
                    );
                  }),
                  e.controlsPrev.forEach(function (n) {
                    return n.removeEventListener(
                      t,
                      e.onNavigatePrevClicked,
                      !1,
                    );
                  }),
                  e.controlsNext.forEach(function (n) {
                    return n.removeEventListener(
                      t,
                      e.onNavigateNextClicked,
                      !1,
                    );
                  });
              });
            },
          },
          {
            key: "update",
            value: function () {
              var e = this.Reveal.availableRoutes();
              []
                .concat(
                  Li(this.controlsLeft),
                  Li(this.controlsRight),
                  Li(this.controlsUp),
                  Li(this.controlsDown),
                  Li(this.controlsPrev),
                  Li(this.controlsNext),
                )
                .forEach(function (e) {
                  e.classList.remove("enabled", "fragmented"),
                    e.setAttribute("disabled", "disabled");
                }),
                e.left &&
                  this.controlsLeft.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  }),
                e.right &&
                  this.controlsRight.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  }),
                e.up &&
                  this.controlsUp.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  }),
                e.down &&
                  this.controlsDown.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  }),
                (e.left || e.up) &&
                  this.controlsPrev.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  }),
                (e.right || e.down) &&
                  this.controlsNext.forEach(function (e) {
                    e.classList.add("enabled"), e.removeAttribute("disabled");
                  });
              var t = this.Reveal.getCurrentSlide();
              if (t) {
                var n = this.Reveal.fragments.availableRoutes();
                n.prev &&
                  this.controlsPrev.forEach(function (e) {
                    e.classList.add("fragmented", "enabled"),
                      e.removeAttribute("disabled");
                  }),
                  n.next &&
                    this.controlsNext.forEach(function (e) {
                      e.classList.add("fragmented", "enabled"),
                        e.removeAttribute("disabled");
                    }),
                  this.Reveal.isVerticalSlide(t)
                    ? (n.prev &&
                        this.controlsUp.forEach(function (e) {
                          e.classList.add("fragmented", "enabled"),
                            e.removeAttribute("disabled");
                        }),
                      n.next &&
                        this.controlsDown.forEach(function (e) {
                          e.classList.add("fragmented", "enabled"),
                            e.removeAttribute("disabled");
                        }))
                    : (n.prev &&
                        this.controlsLeft.forEach(function (e) {
                          e.classList.add("fragmented", "enabled"),
                            e.removeAttribute("disabled");
                        }),
                      n.next &&
                        this.controlsRight.forEach(function (e) {
                          e.classList.add("fragmented", "enabled"),
                            e.removeAttribute("disabled");
                        }));
              }
              if (this.Reveal.getConfig().controlsTutorial) {
                var i = this.Reveal.getIndices();
                !this.Reveal.hasNavigatedVertically() && e.down
                  ? this.controlsDownArrow.classList.add("highlight")
                  : (this.controlsDownArrow.classList.remove("highlight"),
                    this.Reveal.getConfig().rtl
                      ? !this.Reveal.hasNavigatedHorizontally() &&
                        e.left &&
                        0 === i.v
                        ? this.controlsLeftArrow.classList.add("highlight")
                        : this.controlsLeftArrow.classList.remove("highlight")
                      : !this.Reveal.hasNavigatedHorizontally() &&
                        e.right &&
                        0 === i.v
                      ? this.controlsRightArrow.classList.add("highlight")
                      : this.controlsRightArrow.classList.remove("highlight"));
              }
            },
          },
          {
            key: "destroy",
            value: function () {
              this.unbind(), this.element.remove();
            },
          },
          {
            key: "onNavigateLeftClicked",
            value: function (e) {
              e.preventDefault(),
                this.Reveal.onUserInput(),
                "linear" === this.Reveal.getConfig().navigationMode
                  ? this.Reveal.prev()
                  : this.Reveal.left();
            },
          },
          {
            key: "onNavigateRightClicked",
            value: function (e) {
              e.preventDefault(),
                this.Reveal.onUserInput(),
                "linear" === this.Reveal.getConfig().navigationMode
                  ? this.Reveal.next()
                  : this.Reveal.right();
            },
          },
          {
            key: "onNavigateUpClicked",
            value: function (e) {
              e.preventDefault(), this.Reveal.onUserInput(), this.Reveal.up();
            },
          },
          {
            key: "onNavigateDownClicked",
            value: function (e) {
              e.preventDefault(), this.Reveal.onUserInput(), this.Reveal.down();
            },
          },
          {
            key: "onNavigatePrevClicked",
            value: function (e) {
              e.preventDefault(), this.Reveal.onUserInput(), this.Reveal.prev();
            },
          },
          {
            key: "onNavigateNextClicked",
            value: function (e) {
              e.preventDefault(), this.Reveal.onUserInput(), this.Reveal.next();
            },
          },
        ]),
        e
      );
    })(),
    vf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.onProgressClicked = this.onProgressClicked.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "render",
            value: function () {
              (this.element = document.createElement("div")),
                (this.element.className = "progress"),
                this.Reveal.getRevealElement().appendChild(this.element),
                (this.bar = document.createElement("span")),
                this.element.appendChild(this.bar);
            },
          },
          {
            key: "configure",
            value: function (e, t) {
              this.element.style.display = e.progress ? "block" : "none";
            },
          },
          {
            key: "bind",
            value: function () {
              this.Reveal.getConfig().progress &&
                this.element &&
                this.element.addEventListener(
                  "click",
                  this.onProgressClicked,
                  !1,
                );
            },
          },
          {
            key: "unbind",
            value: function () {
              this.Reveal.getConfig().progress &&
                this.element &&
                this.element.removeEventListener(
                  "click",
                  this.onProgressClicked,
                  !1,
                );
            },
          },
          {
            key: "update",
            value: function () {
              if (this.Reveal.getConfig().progress && this.bar) {
                var e = this.Reveal.getProgress();
                this.Reveal.getTotalSlides() < 2 && (e = 0),
                  (this.bar.style.transform = "scaleX(" + e + ")");
              }
            },
          },
          {
            key: "getMaxWidth",
            value: function () {
              return this.Reveal.getRevealElement().offsetWidth;
            },
          },
          {
            key: "onProgressClicked",
            value: function (e) {
              this.Reveal.onUserInput(e), e.preventDefault();
              var t = this.Reveal.getSlides(),
                n = t.length,
                i = Math.floor((e.clientX / this.getMaxWidth()) * n);
              this.Reveal.getConfig().rtl && (i = n - i);
              var r = this.Reveal.getIndices(t[i]);
              this.Reveal.slide(r.h, r.v);
            },
          },
          {
            key: "destroy",
            value: function () {
              this.element.remove();
            },
          },
        ]),
        e
      );
    })(),
    pf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.lastMouseWheelStep = 0),
          (this.cursorHidden = !1),
          (this.cursorInactiveTimeout = 0),
          (this.onDocumentCursorActive =
            this.onDocumentCursorActive.bind(this)),
          (this.onDocumentMouseScroll = this.onDocumentMouseScroll.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "configure",
            value: function (e, t) {
              e.mouseWheel
                ? (document.addEventListener(
                    "DOMMouseScroll",
                    this.onDocumentMouseScroll,
                    !1,
                  ),
                  document.addEventListener(
                    "mousewheel",
                    this.onDocumentMouseScroll,
                    !1,
                  ))
                : (document.removeEventListener(
                    "DOMMouseScroll",
                    this.onDocumentMouseScroll,
                    !1,
                  ),
                  document.removeEventListener(
                    "mousewheel",
                    this.onDocumentMouseScroll,
                    !1,
                  )),
                e.hideInactiveCursor
                  ? (document.addEventListener(
                      "mousemove",
                      this.onDocumentCursorActive,
                      !1,
                    ),
                    document.addEventListener(
                      "mousedown",
                      this.onDocumentCursorActive,
                      !1,
                    ))
                  : (this.showCursor(),
                    document.removeEventListener(
                      "mousemove",
                      this.onDocumentCursorActive,
                      !1,
                    ),
                    document.removeEventListener(
                      "mousedown",
                      this.onDocumentCursorActive,
                      !1,
                    ));
            },
          },
          {
            key: "showCursor",
            value: function () {
              this.cursorHidden &&
                ((this.cursorHidden = !1),
                (this.Reveal.getRevealElement().style.cursor = ""));
            },
          },
          {
            key: "hideCursor",
            value: function () {
              !1 === this.cursorHidden &&
                ((this.cursorHidden = !0),
                (this.Reveal.getRevealElement().style.cursor = "none"));
            },
          },
          {
            key: "destroy",
            value: function () {
              this.showCursor(),
                document.removeEventListener(
                  "DOMMouseScroll",
                  this.onDocumentMouseScroll,
                  !1,
                ),
                document.removeEventListener(
                  "mousewheel",
                  this.onDocumentMouseScroll,
                  !1,
                ),
                document.removeEventListener(
                  "mousemove",
                  this.onDocumentCursorActive,
                  !1,
                ),
                document.removeEventListener(
                  "mousedown",
                  this.onDocumentCursorActive,
                  !1,
                );
            },
          },
          {
            key: "onDocumentCursorActive",
            value: function (e) {
              this.showCursor(),
                clearTimeout(this.cursorInactiveTimeout),
                (this.cursorInactiveTimeout = setTimeout(
                  this.hideCursor.bind(this),
                  this.Reveal.getConfig().hideCursorTime,
                ));
            },
          },
          {
            key: "onDocumentMouseScroll",
            value: function (e) {
              if (Date.now() - this.lastMouseWheelStep > 1e3) {
                this.lastMouseWheelStep = Date.now();
                var t = e.detail || -e.wheelDelta;
                t > 0 ? this.Reveal.next() : t < 0 && this.Reveal.prev();
              }
            },
          },
        ]),
        e
      );
    })(),
    gf = a,
    mf = Jt,
    yf = b,
    bf = o.f,
    wf = function (e) {
      return function (t) {
        for (var n, i = yf(t), r = mf(i), a = r.length, o = 0, s = []; a > o; )
          (n = r[o++]), (gf && !bf.call(i, n)) || s.push(e ? [n, i[n]] : i[n]);
        return s;
      };
    },
    Ef = { entries: wf(!0), values: wf(!1) }.values;
  Xt(
    { target: "Object", stat: !0 },
    {
      values: function (e) {
        return Ef(e);
      },
    },
  );
  var Sf = function (e, t) {
      var n = document.createElement("script");
      (n.type = "text/javascript"),
        (n.async = !1),
        (n.defer = !1),
        (n.src = e),
        "function" == typeof t &&
          ((n.onload = n.onreadystatechange =
            function (e) {
              ("load" === e.type || /loaded|complete/.test(n.readyState)) &&
                ((n.onload = n.onreadystatechange = n.onerror = null), t());
            }),
          (n.onerror = function (e) {
            (n.onload = n.onreadystatechange = n.onerror = null),
              t(new Error("Failed loading script: " + n.src + "\n" + e));
          }));
      var i = document.querySelector("head");
      i.insertBefore(n, i.lastChild);
    },
    kf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.state = "idle"),
          (this.registeredPlugins = {}),
          (this.asyncDependencies = []);
      }
      return (
        Ri(e, [
          {
            key: "load",
            value: function (e, t) {
              var n = this;
              return (
                (this.state = "loading"),
                e.forEach(this.registerPlugin.bind(this)),
                new Promise(function (e) {
                  var i = [],
                    r = 0;
                  if (
                    (t.forEach(function (e) {
                      (e.condition && !e.condition()) ||
                        (e.async ? n.asyncDependencies.push(e) : i.push(e));
                    }),
                    i.length)
                  ) {
                    r = i.length;
                    var a = function (t) {
                      t && "function" == typeof t.callback && t.callback(),
                        0 == --r && n.initPlugins().then(e);
                    };
                    i.forEach(function (e) {
                      "string" == typeof e.id
                        ? (n.registerPlugin(e), a(e))
                        : "string" == typeof e.src
                        ? Sf(e.src, function () {
                            return a(e);
                          })
                        : (console.warn("Unrecognized plugin format", e), a());
                    });
                  } else n.initPlugins().then(e);
                })
              );
            },
          },
          {
            key: "initPlugins",
            value: function () {
              var e = this;
              return new Promise(function (t) {
                var n = Object.values(e.registeredPlugins),
                  i = n.length;
                if (0 === i) e.loadAsync().then(t);
                else {
                  var r,
                    a = function () {
                      0 == --i ? e.loadAsync().then(t) : r();
                    },
                    o = 0;
                  (r = function () {
                    var t = n[o++];
                    if ("function" == typeof t.init) {
                      var i = t.init(e.Reveal);
                      i && "function" == typeof i.then ? i.then(a) : a();
                    } else a();
                  })();
                }
              });
            },
          },
          {
            key: "loadAsync",
            value: function () {
              return (
                (this.state = "loaded"),
                this.asyncDependencies.length &&
                  this.asyncDependencies.forEach(function (e) {
                    Sf(e.src, e.callback);
                  }),
                Promise.resolve()
              );
            },
          },
          {
            key: "registerPlugin",
            value: function (e) {
              2 === arguments.length && "string" == typeof arguments[0]
                ? ((e = arguments[1]).id = arguments[0])
                : "function" == typeof e && (e = e());
              var t = e.id;
              "string" != typeof t
                ? console.warn(
                    "Unrecognized plugin format; can't find plugin.id",
                    e,
                  )
                : void 0 === this.registeredPlugins[t]
                ? ((this.registeredPlugins[t] = e),
                  "loaded" === this.state &&
                    "function" == typeof e.init &&
                    e.init(this.Reveal))
                : console.warn(
                    'reveal.js: "' + t + '" plugin has already been registered',
                  );
            },
          },
          {
            key: "hasPlugin",
            value: function (e) {
              return !!this.registeredPlugins[e];
            },
          },
          {
            key: "getPlugin",
            value: function (e) {
              return this.registeredPlugins[e];
            },
          },
          {
            key: "getRegisteredPlugins",
            value: function () {
              return this.registeredPlugins;
            },
          },
          {
            key: "destroy",
            value: function () {
              Object.values(this.registeredPlugins).forEach(function (e) {
                "function" == typeof e.destroy && e.destroy();
              }),
                (this.registeredPlugins = {}),
                (this.asyncDependencies = []);
            },
          },
        ]),
        e
      );
    })();
  !(function (e) {
    var t = (function (e) {
      var t,
        n = Object.prototype,
        i = n.hasOwnProperty,
        r = "function" == typeof Symbol ? Symbol : {},
        a = r.iterator || "@@iterator",
        o = r.asyncIterator || "@@asyncIterator",
        s = r.toStringTag || "@@toStringTag";
      function l(e, t, n) {
        return (
          Object.defineProperty(e, t, {
            value: n,
            enumerable: !0,
            configurable: !0,
            writable: !0,
          }),
          e[t]
        );
      }
      try {
        l({}, "");
      } catch (e) {
        l = function (e, t, n) {
          return (e[t] = n);
        };
      }
      function c(e, t, n, i) {
        var r = t && t.prototype instanceof g ? t : g,
          a = Object.create(r.prototype),
          o = new P(i || []);
        return (
          (a._invoke = (function (e, t, n) {
            var i = d;
            return function (r, a) {
              if (i === f) throw new Error("Generator is already running");
              if (i === v) {
                if ("throw" === r) throw a;
                return N();
              }
              for (n.method = r, n.arg = a; ; ) {
                var o = n.delegate;
                if (o) {
                  var s = R(o, n);
                  if (s) {
                    if (s === p) continue;
                    return s;
                  }
                }
                if ("next" === n.method) n.sent = n._sent = n.arg;
                else if ("throw" === n.method) {
                  if (i === d) throw ((i = v), n.arg);
                  n.dispatchException(n.arg);
                } else "return" === n.method && n.abrupt("return", n.arg);
                i = f;
                var l = u(e, t, n);
                if ("normal" === l.type) {
                  if (((i = n.done ? v : h), l.arg === p)) continue;
                  return { value: l.arg, done: n.done };
                }
                "throw" === l.type &&
                  ((i = v), (n.method = "throw"), (n.arg = l.arg));
              }
            };
          })(e, n, o)),
          a
        );
      }
      function u(e, t, n) {
        try {
          return { type: "normal", arg: e.call(t, n) };
        } catch (e) {
          return { type: "throw", arg: e };
        }
      }
      e.wrap = c;
      var d = "suspendedStart",
        h = "suspendedYield",
        f = "executing",
        v = "completed",
        p = {};
      function g() {}
      function m() {}
      function y() {}
      var b = {};
      b[a] = function () {
        return this;
      };
      var w = Object.getPrototypeOf,
        E = w && w(w(C([])));
      E && E !== n && i.call(E, a) && (b = E);
      var S = (y.prototype = g.prototype = Object.create(b));
      function k(e) {
        ["next", "throw", "return"].forEach(function (t) {
          l(e, t, function (e) {
            return this._invoke(t, e);
          });
        });
      }
      function A(e, t) {
        function n(r, a, o, s) {
          var l = u(e[r], e, a);
          if ("throw" !== l.type) {
            var c = l.arg,
              d = c.value;
            return d && "object" == typeof d && i.call(d, "__await")
              ? t.resolve(d.__await).then(
                  function (e) {
                    n("next", e, o, s);
                  },
                  function (e) {
                    n("throw", e, o, s);
                  },
                )
              : t.resolve(d).then(
                  function (e) {
                    (c.value = e), o(c);
                  },
                  function (e) {
                    return n("throw", e, o, s);
                  },
                );
          }
          s(l.arg);
        }
        var r;
        this._invoke = function (e, i) {
          function a() {
            return new t(function (t, r) {
              n(e, i, t, r);
            });
          }
          return (r = r ? r.then(a, a) : a());
        };
      }
      function R(e, n) {
        var i = e.iterator[n.method];
        if (i === t) {
          if (((n.delegate = null), "throw" === n.method)) {
            if (
              e.iterator.return &&
              ((n.method = "return"),
              (n.arg = t),
              R(e, n),
              "throw" === n.method)
            )
              return p;
            (n.method = "throw"),
              (n.arg = new TypeError(
                "The iterator does not provide a 'throw' method",
              ));
          }
          return p;
        }
        var r = u(i, e.iterator, n.arg);
        if ("throw" === r.type)
          return (n.method = "throw"), (n.arg = r.arg), (n.delegate = null), p;
        var a = r.arg;
        return a
          ? a.done
            ? ((n[e.resultName] = a.value),
              (n.next = e.nextLoc),
              "return" !== n.method && ((n.method = "next"), (n.arg = t)),
              (n.delegate = null),
              p)
            : a
          : ((n.method = "throw"),
            (n.arg = new TypeError("iterator result is not an object")),
            (n.delegate = null),
            p);
      }
      function x(e) {
        var t = { tryLoc: e[0] };
        1 in e && (t.catchLoc = e[1]),
          2 in e && ((t.finallyLoc = e[2]), (t.afterLoc = e[3])),
          this.tryEntries.push(t);
      }
      function L(e) {
        var t = e.completion || {};
        (t.type = "normal"), delete t.arg, (e.completion = t);
      }
      function P(e) {
        (this.tryEntries = [{ tryLoc: "root" }]),
          e.forEach(x, this),
          this.reset(!0);
      }
      function C(e) {
        if (e) {
          var n = e[a];
          if (n) return n.call(e);
          if ("function" == typeof e.next) return e;
          if (!isNaN(e.length)) {
            var r = -1,
              o = function n() {
                for (; ++r < e.length; )
                  if (i.call(e, r)) return (n.value = e[r]), (n.done = !1), n;
                return (n.value = t), (n.done = !0), n;
              };
            return (o.next = o);
          }
        }
        return { next: N };
      }
      function N() {
        return { value: t, done: !0 };
      }
      return (
        (m.prototype = S.constructor = y),
        (y.constructor = m),
        (m.displayName = l(y, s, "GeneratorFunction")),
        (e.isGeneratorFunction = function (e) {
          var t = "function" == typeof e && e.constructor;
          return (
            !!t &&
            (t === m || "GeneratorFunction" === (t.displayName || t.name))
          );
        }),
        (e.mark = function (e) {
          return (
            Object.setPrototypeOf
              ? Object.setPrototypeOf(e, y)
              : ((e.__proto__ = y), l(e, s, "GeneratorFunction")),
            (e.prototype = Object.create(S)),
            e
          );
        }),
        (e.awrap = function (e) {
          return { __await: e };
        }),
        k(A.prototype),
        (A.prototype[o] = function () {
          return this;
        }),
        (e.AsyncIterator = A),
        (e.async = function (t, n, i, r, a) {
          void 0 === a && (a = Promise);
          var o = new A(c(t, n, i, r), a);
          return e.isGeneratorFunction(n)
            ? o
            : o.next().then(function (e) {
                return e.done ? e.value : o.next();
              });
        }),
        k(S),
        l(S, s, "Generator"),
        (S[a] = function () {
          return this;
        }),
        (S.toString = function () {
          return "[object Generator]";
        }),
        (e.keys = function (e) {
          var t = [];
          for (var n in e) t.push(n);
          return (
            t.reverse(),
            function n() {
              for (; t.length; ) {
                var i = t.pop();
                if (i in e) return (n.value = i), (n.done = !1), n;
              }
              return (n.done = !0), n;
            }
          );
        }),
        (e.values = C),
        (P.prototype = {
          constructor: P,
          reset: function (e) {
            if (
              ((this.prev = 0),
              (this.next = 0),
              (this.sent = this._sent = t),
              (this.done = !1),
              (this.delegate = null),
              (this.method = "next"),
              (this.arg = t),
              this.tryEntries.forEach(L),
              !e)
            )
              for (var n in this)
                "t" === n.charAt(0) &&
                  i.call(this, n) &&
                  !isNaN(+n.slice(1)) &&
                  (this[n] = t);
          },
          stop: function () {
            this.done = !0;
            var e = this.tryEntries[0].completion;
            if ("throw" === e.type) throw e.arg;
            return this.rval;
          },
          dispatchException: function (e) {
            if (this.done) throw e;
            var n = this;
            function r(i, r) {
              return (
                (s.type = "throw"),
                (s.arg = e),
                (n.next = i),
                r && ((n.method = "next"), (n.arg = t)),
                !!r
              );
            }
            for (var a = this.tryEntries.length - 1; a >= 0; --a) {
              var o = this.tryEntries[a],
                s = o.completion;
              if ("root" === o.tryLoc) return r("end");
              if (o.tryLoc <= this.prev) {
                var l = i.call(o, "catchLoc"),
                  c = i.call(o, "finallyLoc");
                if (l && c) {
                  if (this.prev < o.catchLoc) return r(o.catchLoc, !0);
                  if (this.prev < o.finallyLoc) return r(o.finallyLoc);
                } else if (l) {
                  if (this.prev < o.catchLoc) return r(o.catchLoc, !0);
                } else {
                  if (!c)
                    throw new Error("try statement without catch or finally");
                  if (this.prev < o.finallyLoc) return r(o.finallyLoc);
                }
              }
            }
          },
          abrupt: function (e, t) {
            for (var n = this.tryEntries.length - 1; n >= 0; --n) {
              var r = this.tryEntries[n];
              if (
                r.tryLoc <= this.prev &&
                i.call(r, "finallyLoc") &&
                this.prev < r.finallyLoc
              ) {
                var a = r;
                break;
              }
            }
            a &&
              ("break" === e || "continue" === e) &&
              a.tryLoc <= t &&
              t <= a.finallyLoc &&
              (a = null);
            var o = a ? a.completion : {};
            return (
              (o.type = e),
              (o.arg = t),
              a
                ? ((this.method = "next"), (this.next = a.finallyLoc), p)
                : this.complete(o)
            );
          },
          complete: function (e, t) {
            if ("throw" === e.type) throw e.arg;
            return (
              "break" === e.type || "continue" === e.type
                ? (this.next = e.arg)
                : "return" === e.type
                ? ((this.rval = this.arg = e.arg),
                  (this.method = "return"),
                  (this.next = "end"))
                : "normal" === e.type && t && (this.next = t),
              p
            );
          },
          finish: function (e) {
            for (var t = this.tryEntries.length - 1; t >= 0; --t) {
              var n = this.tryEntries[t];
              if (n.finallyLoc === e)
                return this.complete(n.completion, n.afterLoc), L(n), p;
            }
          },
          catch: function (e) {
            for (var t = this.tryEntries.length - 1; t >= 0; --t) {
              var n = this.tryEntries[t];
              if (n.tryLoc === e) {
                var i = n.completion;
                if ("throw" === i.type) {
                  var r = i.arg;
                  L(n);
                }
                return r;
              }
            }
            throw new Error("illegal catch attempt");
          },
          delegateYield: function (e, n, i) {
            return (
              (this.delegate = { iterator: C(e), resultName: n, nextLoc: i }),
              "next" === this.method && (this.arg = t),
              p
            );
          },
        }),
        e
      );
    })(e.exports);
    try {
      regeneratorRuntime = t;
    } catch (e) {
      Function("r", "regeneratorRuntime = r")(t);
    }
  })({ exports: {} });
  var Af = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      var t, n;
      return (
        Ri(e, [
          {
            key: "setupPDF",
            value:
              ((t = regeneratorRuntime.mark(function e() {
                var t, n, i, r, a, o, s, l, c, u, d, h, f, v;
                return regeneratorRuntime.wrap(
                  function (e) {
                    for (;;)
                      switch ((e.prev = e.next)) {
                        case 0:
                          return (
                            (t = this.Reveal.getConfig()),
                            (n = yh(this.Reveal.getRevealElement(), tf)),
                            (i =
                              t.slideNumber &&
                              /all|print/i.test(t.showSlideNumber)),
                            (r = this.Reveal.getComputedSlideSize(
                              window.innerWidth,
                              window.innerHeight,
                            )),
                            (a = Math.floor(r.width * (1 + t.margin))),
                            (o = Math.floor(r.height * (1 + t.margin))),
                            (s = r.width),
                            (l = r.height),
                            (e.next = 8),
                            new Promise(requestAnimationFrame)
                          );
                        case 8:
                          return (
                            Rh(
                              "@page{size:" +
                                a +
                                "px " +
                                o +
                                "px; margin: 0px;}",
                            ),
                            Rh(
                              ".reveal section>img, .reveal section>video, .reveal section>iframe{max-width: " +
                                s +
                                "px; max-height:" +
                                l +
                                "px}",
                            ),
                            document.documentElement.classList.add("print-pdf"),
                            (document.body.style.width = a + "px"),
                            (document.body.style.height = o + "px"),
                            (c = document.querySelector(".reveal-viewport")) &&
                              (d = window.getComputedStyle(c)) &&
                              d.background &&
                              (u = d.background),
                            (e.next = 17),
                            new Promise(requestAnimationFrame)
                          );
                        case 17:
                          return (
                            this.Reveal.layoutSlideContents(s, l),
                            (e.next = 20),
                            new Promise(requestAnimationFrame)
                          );
                        case 20:
                          return (
                            (h = n.map(function (e) {
                              return e.scrollHeight;
                            })),
                            (f = []),
                            (v = n[0].parentNode),
                            n.forEach(function (e, n) {
                              if (!1 === e.classList.contains("stack")) {
                                var r = (a - s) / 2,
                                  c = (o - l) / 2,
                                  d = h[n],
                                  v = Math.max(Math.ceil(d / o), 1);
                                ((1 ===
                                  (v = Math.min(v, t.pdfMaxPagesPerSlide)) &&
                                  t.center) ||
                                  e.classList.contains("center")) &&
                                  (c = Math.max((o - d) / 2, 0));
                                var p = document.createElement("div");
                                if (
                                  (f.push(p),
                                  (p.className = "pdf-page"),
                                  (p.style.height =
                                    (o + t.pdfPageHeightOffset) * v + "px"),
                                  u && (p.style.background = u),
                                  p.appendChild(e),
                                  (e.style.left = r + "px"),
                                  (e.style.top = c + "px"),
                                  (e.style.width = s + "px"),
                                  this.Reveal.slideContent.layout(e),
                                  e.slideBackgroundElement &&
                                    p.insertBefore(e.slideBackgroundElement, e),
                                  t.showNotes)
                                ) {
                                  var g = this.Reveal.getSlideNotes(e);
                                  if (g) {
                                    var m =
                                        "string" == typeof t.showNotes
                                          ? t.showNotes
                                          : "inline",
                                      y = document.createElement("div");
                                    y.classList.add("speaker-notes"),
                                      y.classList.add("speaker-notes-pdf"),
                                      y.setAttribute("data-layout", m),
                                      (y.innerHTML = g),
                                      "separate-page" === m
                                        ? f.push(y)
                                        : ((y.style.left = "8px"),
                                          (y.style.bottom = "8px"),
                                          (y.style.width = a - 16 + "px"),
                                          p.appendChild(y));
                                  }
                                }
                                if (i) {
                                  var b = n + 1,
                                    w = document.createElement("div");
                                  w.classList.add("slide-number"),
                                    w.classList.add("slide-number-pdf"),
                                    (w.innerHTML = b),
                                    p.appendChild(w);
                                }
                                if (t.pdfSeparateFragments) {
                                  var E,
                                    S = this.Reveal.fragments.sort(
                                      p.querySelectorAll(".fragment"),
                                      !0,
                                    );
                                  S.forEach(function (e) {
                                    E &&
                                      E.forEach(function (e) {
                                        e.classList.remove("current-fragment");
                                      }),
                                      e.forEach(function (e) {
                                        e.classList.add(
                                          "visible",
                                          "current-fragment",
                                        );
                                      }, this);
                                    var t = p.cloneNode(!0);
                                    f.push(t), (E = e);
                                  }, this),
                                    S.forEach(function (e) {
                                      e.forEach(function (e) {
                                        e.classList.remove(
                                          "visible",
                                          "current-fragment",
                                        );
                                      });
                                    });
                                } else
                                  yh(p, ".fragment:not(.fade-out)").forEach(
                                    function (e) {
                                      e.classList.add("visible");
                                    },
                                  );
                              }
                            }, this),
                            (e.next = 26),
                            new Promise(requestAnimationFrame)
                          );
                        case 26:
                          f.forEach(function (e) {
                            return v.appendChild(e);
                          }),
                            this.Reveal.dispatchEvent({ type: "pdf-ready" });
                        case 28:
                        case "end":
                          return e.stop();
                      }
                  },
                  e,
                  this,
                );
              })),
              (n = function () {
                var e = this,
                  n = arguments;
                return new Promise(function (i, r) {
                  var a = t.apply(e, n);
                  function o(e) {
                    Si(a, i, r, o, s, "next", e);
                  }
                  function s(e) {
                    Si(a, i, r, o, s, "throw", e);
                  }
                  o(void 0);
                });
              }),
              function () {
                return n.apply(this, arguments);
              }),
          },
          {
            key: "isPrintingPDF",
            value: function () {
              return /print-pdf/gi.test(window.location.search);
            },
          },
        ]),
        e
      );
    })(),
    Rf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.touchStartX = 0),
          (this.touchStartY = 0),
          (this.touchStartCount = 0),
          (this.touchCaptured = !1),
          (this.onPointerDown = this.onPointerDown.bind(this)),
          (this.onPointerMove = this.onPointerMove.bind(this)),
          (this.onPointerUp = this.onPointerUp.bind(this)),
          (this.onTouchStart = this.onTouchStart.bind(this)),
          (this.onTouchMove = this.onTouchMove.bind(this)),
          (this.onTouchEnd = this.onTouchEnd.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "bind",
            value: function () {
              var e = this.Reveal.getRevealElement();
              "onpointerdown" in window
                ? (e.addEventListener("pointerdown", this.onPointerDown, !1),
                  e.addEventListener("pointermove", this.onPointerMove, !1),
                  e.addEventListener("pointerup", this.onPointerUp, !1))
                : window.navigator.msPointerEnabled
                ? (e.addEventListener("MSPointerDown", this.onPointerDown, !1),
                  e.addEventListener("MSPointerMove", this.onPointerMove, !1),
                  e.addEventListener("MSPointerUp", this.onPointerUp, !1))
                : (e.addEventListener("touchstart", this.onTouchStart, !1),
                  e.addEventListener("touchmove", this.onTouchMove, !1),
                  e.addEventListener("touchend", this.onTouchEnd, !1));
            },
          },
          {
            key: "unbind",
            value: function () {
              var e = this.Reveal.getRevealElement();
              e.removeEventListener("pointerdown", this.onPointerDown, !1),
                e.removeEventListener("pointermove", this.onPointerMove, !1),
                e.removeEventListener("pointerup", this.onPointerUp, !1),
                e.removeEventListener("MSPointerDown", this.onPointerDown, !1),
                e.removeEventListener("MSPointerMove", this.onPointerMove, !1),
                e.removeEventListener("MSPointerUp", this.onPointerUp, !1),
                e.removeEventListener("touchstart", this.onTouchStart, !1),
                e.removeEventListener("touchmove", this.onTouchMove, !1),
                e.removeEventListener("touchend", this.onTouchEnd, !1);
            },
          },
          {
            key: "isSwipePrevented",
            value: function (e) {
              if (Sh(e, "video, audio")) return !0;
              for (; e && "function" == typeof e.hasAttribute; ) {
                if (e.hasAttribute("data-prevent-swipe")) return !0;
                e = e.parentNode;
              }
              return !1;
            },
          },
          {
            key: "onTouchStart",
            value: function (e) {
              if (this.isSwipePrevented(e.target)) return !0;
              (this.touchStartX = e.touches[0].clientX),
                (this.touchStartY = e.touches[0].clientY),
                (this.touchStartCount = e.touches.length);
            },
          },
          {
            key: "onTouchMove",
            value: function (e) {
              if (this.isSwipePrevented(e.target)) return !0;
              var t = this.Reveal.getConfig();
              if (this.touchCaptured) Ih && e.preventDefault();
              else {
                this.Reveal.onUserInput(e);
                var n = e.touches[0].clientX,
                  i = e.touches[0].clientY;
                if (1 === e.touches.length && 2 !== this.touchStartCount) {
                  var r = this.Reveal.availableRoutes({ includeFragments: !0 }),
                    a = n - this.touchStartX,
                    o = i - this.touchStartY;
                  a > 40 && Math.abs(a) > Math.abs(o)
                    ? ((this.touchCaptured = !0),
                      "linear" === t.navigationMode
                        ? t.rtl
                          ? this.Reveal.next()
                          : this.Reveal.prev()
                        : this.Reveal.left())
                    : a < -40 && Math.abs(a) > Math.abs(o)
                    ? ((this.touchCaptured = !0),
                      "linear" === t.navigationMode
                        ? t.rtl
                          ? this.Reveal.prev()
                          : this.Reveal.next()
                        : this.Reveal.right())
                    : o > 40 && r.up
                    ? ((this.touchCaptured = !0),
                      "linear" === t.navigationMode
                        ? this.Reveal.prev()
                        : this.Reveal.up())
                    : o < -40 &&
                      r.down &&
                      ((this.touchCaptured = !0),
                      "linear" === t.navigationMode
                        ? this.Reveal.next()
                        : this.Reveal.down()),
                    t.embedded
                      ? (this.touchCaptured || this.Reveal.isVerticalSlide()) &&
                        e.preventDefault()
                      : e.preventDefault();
                }
              }
            },
          },
          {
            key: "onTouchEnd",
            value: function (e) {
              this.touchCaptured = !1;
            },
          },
          {
            key: "onPointerDown",
            value: function (e) {
              (e.pointerType !== e.MSPOINTER_TYPE_TOUCH &&
                "touch" !== e.pointerType) ||
                ((e.touches = [{ clientX: e.clientX, clientY: e.clientY }]),
                this.onTouchStart(e));
            },
          },
          {
            key: "onPointerMove",
            value: function (e) {
              (e.pointerType !== e.MSPOINTER_TYPE_TOUCH &&
                "touch" !== e.pointerType) ||
                ((e.touches = [{ clientX: e.clientX, clientY: e.clientY }]),
                this.onTouchMove(e));
            },
          },
          {
            key: "onPointerUp",
            value: function (e) {
              (e.pointerType !== e.MSPOINTER_TYPE_TOUCH &&
                "touch" !== e.pointerType) ||
                ((e.touches = [{ clientX: e.clientX, clientY: e.clientY }]),
                this.onTouchEnd(e));
            },
          },
        ]),
        e
      );
    })(),
    xf = "focus",
    Lf = "blur",
    Pf = (function () {
      function e(t) {
        ki(this, e),
          (this.Reveal = t),
          (this.onRevealPointerDown = this.onRevealPointerDown.bind(this)),
          (this.onDocumentPointerDown = this.onDocumentPointerDown.bind(this));
      }
      return (
        Ri(e, [
          {
            key: "configure",
            value: function (e, t) {
              e.embedded ? this.blur() : (this.focus(), this.unbind());
            },
          },
          {
            key: "bind",
            value: function () {
              this.Reveal.getConfig().embedded &&
                this.Reveal.getRevealElement().addEventListener(
                  "pointerdown",
                  this.onRevealPointerDown,
                  !1,
                );
            },
          },
          {
            key: "unbind",
            value: function () {
              this.Reveal.getRevealElement().removeEventListener(
                "pointerdown",
                this.onRevealPointerDown,
                !1,
              ),
                document.removeEventListener(
                  "pointerdown",
                  this.onDocumentPointerDown,
                  !1,
                );
            },
          },
          {
            key: "focus",
            value: function () {
              this.state !== xf &&
                (this.Reveal.getRevealElement().classList.add("focused"),
                document.addEventListener(
                  "pointerdown",
                  this.onDocumentPointerDown,
                  !1,
                )),
                (this.state = xf);
            },
          },
          {
            key: "blur",
            value: function () {
              this.state !== Lf &&
                (this.Reveal.getRevealElement().classList.remove("focused"),
                document.removeEventListener(
                  "pointerdown",
                  this.onDocumentPointerDown,
                  !1,
                )),
                (this.state = Lf);
            },
          },
          {
            key: "isFocused",
            value: function () {
              return this.state === xf;
            },
          },
          {
            key: "destroy",
            value: function () {
              this.Reveal.getRevealElement().classList.remove("focused");
            },
          },
          {
            key: "onRevealPointerDown",
            value: function (e) {
              this.focus();
            },
          },
          {
            key: "onDocumentPointerDown",
            value: function (e) {
              var t = kh(e.target, ".reveal");
              (t && t === this.Reveal.getRevealElement()) || this.blur();
            },
          },
        ]),
        e
      );
    })(),
    Cf = (function () {
      function e(t) {
        ki(this, e), (this.Reveal = t);
      }
      return (
        Ri(e, [
          {
            key: "render",
            value: function () {
              (this.element = document.createElement("div")),
                (this.element.className = "speaker-notes"),
                this.element.setAttribute("data-prevent-swipe", ""),
                this.element.setAttribute("tabindex", "0"),
                this.Reveal.getRevealElement().appendChild(this.element);
            },
          },
          {
            key: "configure",
            value: function (e, t) {
              e.showNotes &&
                this.element.setAttribute(
                  "data-layout",
                  "string" == typeof e.showNotes ? e.showNotes : "inline",
                );
            },
          },
          {
            key: "update",
            value: function () {
              this.Reveal.getConfig().showNotes &&
                this.element &&
                this.Reveal.getCurrentSlide() &&
                !this.Reveal.print.isPrintingPDF() &&
                (this.element.innerHTML =
                  this.getSlideNotes() ||
                  '<span class="notes-placeholder">No notes on this slide.</span>');
            },
          },
          {
            key: "updateVisibility",
            value: function () {
              this.Reveal.getConfig().showNotes &&
              this.hasNotes() &&
              !this.Reveal.print.isPrintingPDF()
                ? this.Reveal.getRevealElement().classList.add("show-notes")
                : this.Reveal.getRevealElement().classList.remove("show-notes");
            },
          },
          {
            key: "hasNotes",
            value: function () {
              return (
                this.Reveal.getSlidesElement().querySelectorAll(
                  "[data-notes], aside.notes",
                ).length > 0
              );
            },
          },
          {
            key: "isSpeakerNotesWindow",
            value: function () {
              return !!window.location.search.match(/receiver/gi);
            },
          },
          {
            key: "getSlideNotes",
            value: function () {
              var e =
                arguments.length > 0 && void 0 !== arguments[0]
                  ? arguments[0]
                  : this.Reveal.getCurrentSlide();
              if (e.hasAttribute("data-notes"))
                return e.getAttribute("data-notes");
              var t = e.querySelector("aside.notes");
              return t ? t.innerHTML : null;
            },
          },
          {
            key: "destroy",
            value: function () {
              this.element.remove();
            },
          },
        ]),
        e
      );
    })(),
    Nf = A,
    Mf = ct,
    If = at,
    Tf = Hs,
    Of = B,
    Df = Dn("unscopables"),
    jf = Array.prototype;
  null == jf[Df] && Of.f(jf, Df, { configurable: !0, value: Tf(null) });
  var Ff = function (e) {
    jf[Df][e] = !0;
  };
  Xt(
    { target: "Array", proto: !0 },
    {
      fill: function (e) {
        for (
          var t = Nf(this),
            n = If(t.length),
            i = arguments.length,
            r = Mf(i > 1 ? arguments[1] : void 0, n),
            a = i > 2 ? arguments[2] : void 0,
            o = void 0 === a ? n : Mf(a, n);
          o > r;

        )
          t[r++] = e;
        return t;
      },
    },
  ),
    Ff("fill");
  var zf = (function () {
      function e(t, n) {
        ki(this, e),
          (this.diameter = 100),
          (this.diameter2 = this.diameter / 2),
          (this.thickness = 6),
          (this.playing = !1),
          (this.progress = 0),
          (this.progressOffset = 1),
          (this.container = t),
          (this.progressCheck = n),
          (this.canvas = document.createElement("canvas")),
          (this.canvas.className = "playback"),
          (this.canvas.width = this.diameter),
          (this.canvas.height = this.diameter),
          (this.canvas.style.width = this.diameter2 + "px"),
          (this.canvas.style.height = this.diameter2 + "px"),
          (this.context = this.canvas.getContext("2d")),
          this.container.appendChild(this.canvas),
          this.render();
      }
      return (
        Ri(e, [
          {
            key: "setPlaying",
            value: function (e) {
              var t = this.playing;
              (this.playing = e),
                !t && this.playing ? this.animate() : this.render();
            },
          },
          {
            key: "animate",
            value: function () {
              var e = this.progress;
              (this.progress = this.progressCheck()),
                e > 0.8 &&
                  this.progress < 0.2 &&
                  (this.progressOffset = this.progress),
                this.render(),
                this.playing && requestAnimationFrame(this.animate.bind(this));
            },
          },
          {
            key: "render",
            value: function () {
              var e = this.playing ? this.progress : 0,
                t = this.diameter2 - this.thickness,
                n = this.diameter2,
                i = this.diameter2,
                r = 28;
              this.progressOffset += 0.1 * (1 - this.progressOffset);
              var a = -Math.PI / 2 + e * (2 * Math.PI),
                o = -Math.PI / 2 + this.progressOffset * (2 * Math.PI);
              this.context.save(),
                this.context.clearRect(0, 0, this.diameter, this.diameter),
                this.context.beginPath(),
                this.context.arc(n, i, t + 4, 0, 2 * Math.PI, !1),
                (this.context.fillStyle = "rgba( 0, 0, 0, 0.4 )"),
                this.context.fill(),
                this.context.beginPath(),
                this.context.arc(n, i, t, 0, 2 * Math.PI, !1),
                (this.context.lineWidth = this.thickness),
                (this.context.strokeStyle = "rgba( 255, 255, 255, 0.2 )"),
                this.context.stroke(),
                this.playing &&
                  (this.context.beginPath(),
                  this.context.arc(n, i, t, o, a, !1),
                  (this.context.lineWidth = this.thickness),
                  (this.context.strokeStyle = "#fff"),
                  this.context.stroke()),
                this.context.translate(n - 14, i - 14),
                this.playing
                  ? ((this.context.fillStyle = "#fff"),
                    this.context.fillRect(0, 0, 10, r),
                    this.context.fillRect(18, 0, 10, r))
                  : (this.context.beginPath(),
                    this.context.translate(4, 0),
                    this.context.moveTo(0, 0),
                    this.context.lineTo(24, 14),
                    this.context.lineTo(0, r),
                    (this.context.fillStyle = "#fff"),
                    this.context.fill()),
                this.context.restore();
            },
          },
          {
            key: "on",
            value: function (e, t) {
              this.canvas.addEventListener(e, t, !1);
            },
          },
          {
            key: "off",
            value: function (e, t) {
              this.canvas.removeEventListener(e, t, !1);
            },
          },
          {
            key: "destroy",
            value: function () {
              (this.playing = !1),
                this.canvas.parentNode &&
                  this.container.removeChild(this.canvas);
            },
          },
        ]),
        e
      );
    })(),
    Hf = {
      width: 960,
      height: 700,
      margin: 0.04,
      minScale: 0.2,
      maxScale: 2,
      controls: !0,
      controlsTutorial: !0,
      controlsLayout: "bottom-right",
      controlsBackArrows: "faded",
      progress: !0,
      slideNumber: !1,
      showSlideNumber: "all",
      hashOneBasedIndex: !1,
      hash: !1,
      respondToHashChanges: !0,
      history: !1,
      keyboard: !0,
      keyboardCondition: null,
      disableLayout: !1,
      overview: !0,
      center: !0,
      touch: !0,
      loop: !1,
      rtl: !1,
      navigationMode: "default",
      shuffle: !1,
      fragments: !0,
      fragmentInURL: !0,
      embedded: !1,
      help: !0,
      pause: !0,
      showNotes: !1,
      showHiddenSlides: !1,
      autoPlayMedia: null,
      preloadIframes: null,
      autoAnimate: !0,
      autoAnimateMatcher: null,
      autoAnimateEasing: "ease",
      autoAnimateDuration: 1,
      autoAnimateUnmatched: !0,
      autoAnimateStyles: [
        "opacity",
        "color",
        "background-color",
        "padding",
        "font-size",
        "line-height",
        "letter-spacing",
        "border-width",
        "border-color",
        "border-radius",
        "outline",
        "outline-offset",
      ],
      autoSlide: 0,
      autoSlideStoppable: !0,
      autoSlideMethod: null,
      defaultTiming: null,
      mouseWheel: !1,
      previewLinks: !1,
      postMessage: !0,
      postMessageEvents: !1,
      focusBodyOnPageVisibilityChange: !0,
      transition: "slide",
      transitionSpeed: "default",
      backgroundTransition: "fade",
      parallaxBackgroundImage: "",
      parallaxBackgroundSize: "",
      parallaxBackgroundRepeat: "",
      parallaxBackgroundPosition: "",
      parallaxBackgroundHorizontal: null,
      parallaxBackgroundVertical: null,
      pdfMaxPagesPerSlide: Number.POSITIVE_INFINITY,
      pdfSeparateFragments: !0,
      pdfPageHeightOffset: -1,
      viewDistance: 3,
      mobileViewDistance: 2,
      display: "block",
      hideInactiveCursor: !0,
      hideCursorTime: 5e3,
      dependencies: [],
      plugins: [],
    },
    Uf = "4.3.1";
  function _f(e, t) {
    arguments.length < 2 &&
      ((t = arguments[0]), (e = document.querySelector(".reveal")));
    var n,
      i,
      r,
      a,
      o,
      s = {},
      l = {},
      c = !1,
      u = { hasNavigatedHorizontally: !1, hasNavigatedVertically: !1 },
      d = [],
      h = 1,
      f = { layout: "", overview: "" },
      v = {},
      p = "idle",
      g = 0,
      m = 0,
      y = -1,
      b = !1,
      w = new jh(s),
      E = new Fh(s),
      S = new lf(s),
      k = new Jh(s),
      A = new cf(s),
      R = new uf(s),
      x = new df(s),
      L = new hf(s),
      P = new ff(s),
      C = new vf(s),
      N = new pf(s),
      M = new kf(s),
      I = new Af(s),
      T = new Pf(s),
      O = new Rf(s),
      D = new Cf(s);
    function j(n) {
      if (!e) throw 'Unable to find presentation root (<div class="reveal">).';
      if (((v.wrapper = e), (v.slides = e.querySelector(".slides")), !v.slides))
        throw 'Unable to find slides container (<div class="slides">).';
      return (
        (l = wi(wi(wi(wi(wi({}, Hf), l), t), n), xh())),
        F(),
        window.addEventListener("load", le, !1),
        M.load(l.plugins, l.dependencies).then(z),
        new Promise(function (e) {
          return s.on("ready", e);
        })
      );
    }
    function F() {
      !0 === l.embedded
        ? (v.viewport = kh(e, ".reveal-viewport") || e)
        : ((v.viewport = document.body),
          document.documentElement.classList.add("reveal-full-page")),
        v.viewport.classList.add("reveal-viewport");
    }
    function z() {
      (c = !0),
        H(),
        U(),
        K(),
        W(),
        V(),
        xe(),
        Y(),
        L.readURL(),
        k.update(!0),
        setTimeout(function () {
          v.slides.classList.remove("no-transition"),
            v.wrapper.classList.add("ready"),
            ee({
              type: "ready",
              data: { indexh: n, indexv: i, currentSlide: a },
            });
        }, 1),
        I.isPrintingPDF() &&
          ($(),
          "complete" === document.readyState
            ? I.setupPDF()
            : window.addEventListener("load", function () {
                I.setupPDF();
              }));
    }
    function H() {
      l.showHiddenSlides ||
        yh(v.wrapper, 'section[data-visibility="hidden"]').forEach(
          function (e) {
            e.parentNode.removeChild(e);
          },
        );
    }
    function U() {
      v.slides.classList.add("no-transition"),
        Mh
          ? v.wrapper.classList.add("no-hover")
          : v.wrapper.classList.remove("no-hover"),
        k.render(),
        E.render(),
        P.render(),
        C.render(),
        D.render(),
        (v.pauseOverlay = Ah(
          v.wrapper,
          "div",
          "pause-overlay",
          l.controls
            ? '<button class="resume-button">Resume presentation</button>'
            : null,
        )),
        (v.statusElement = _()),
        v.wrapper.setAttribute("role", "application");
    }
    function _() {
      var e = v.wrapper.querySelector(".aria-status");
      return (
        e ||
          (((e = document.createElement("div")).style.position = "absolute"),
          (e.style.height = "1px"),
          (e.style.width = "1px"),
          (e.style.overflow = "hidden"),
          (e.style.clip = "rect( 1px, 1px, 1px, 1px )"),
          e.classList.add("aria-status"),
          e.setAttribute("aria-live", "polite"),
          e.setAttribute("aria-atomic", "true"),
          v.wrapper.appendChild(e)),
        e
      );
    }
    function B(e) {
      v.statusElement.textContent = e;
    }
    function q(e) {
      var t = "";
      if (3 === e.nodeType) t += e.textContent;
      else if (1 === e.nodeType) {
        var n = e.getAttribute("aria-hidden"),
          i = "none" === window.getComputedStyle(e).display;
        "true" === n ||
          i ||
          Array.from(e.childNodes).forEach(function (e) {
            t += q(e);
          });
      }
      return "" === (t = t.trim()) ? "" : t + " ";
    }
    function W() {
      setInterval(function () {
        (0 === v.wrapper.scrollTop && 0 === v.wrapper.scrollLeft) ||
          ((v.wrapper.scrollTop = 0), (v.wrapper.scrollLeft = 0));
      }, 1e3);
    }
    function V() {
      document.addEventListener("fullscreenchange", lt),
        document.addEventListener("webkitfullscreenchange", lt);
    }
    function K() {
      l.postMessage && window.addEventListener("message", it, !1);
    }
    function Y(e) {
      var t = wi({}, l);
      if (("object" === Ei(e) && mh(l, e), !1 !== s.isReady())) {
        var n = v.wrapper.querySelectorAll(tf).length;
        v.wrapper.classList.remove(t.transition),
          v.wrapper.classList.add(l.transition),
          v.wrapper.setAttribute("data-transition-speed", l.transitionSpeed),
          v.wrapper.setAttribute(
            "data-background-transition",
            l.backgroundTransition,
          ),
          v.viewport.style.setProperty("--slide-width", l.width + "px"),
          v.viewport.style.setProperty("--slide-height", l.height + "px"),
          l.shuffle && Le(),
          bh(v.wrapper, "embedded", l.embedded),
          bh(v.wrapper, "rtl", l.rtl),
          bh(v.wrapper, "center", l.center),
          !1 === l.pause && ye(),
          l.previewLinks
            ? (ne(), ie("[data-preview-link=false]"))
            : (ie(), ne("[data-preview-link]:not([data-preview-link=false])")),
          S.reset(),
          o && (o.destroy(), (o = null)),
          n > 1 &&
            l.autoSlide &&
            l.autoSlideStoppable &&
            ((o = new zf(v.wrapper, function () {
              return Math.min(Math.max((Date.now() - y) / g, 0), 1);
            })).on("click", ut),
            (b = !1)),
          "default" !== l.navigationMode
            ? v.wrapper.setAttribute("data-navigation-mode", l.navigationMode)
            : v.wrapper.removeAttribute("data-navigation-mode"),
          D.configure(l, t),
          T.configure(l, t),
          N.configure(l, t),
          P.configure(l, t),
          C.configure(l, t),
          x.configure(l, t),
          A.configure(l, t),
          E.configure(l, t),
          Ae();
      }
    }
    function X() {
      window.addEventListener("resize", ot, !1),
        l.touch && O.bind(),
        l.keyboard && x.bind(),
        l.progress && C.bind(),
        l.respondToHashChanges && L.bind(),
        P.bind(),
        T.bind(),
        v.slides.addEventListener("click", at, !1),
        v.slides.addEventListener("transitionend", rt, !1),
        v.pauseOverlay.addEventListener("click", ye, !1),
        l.focusBodyOnPageVisibilityChange &&
          document.addEventListener("visibilitychange", st, !1);
    }
    function $() {
      O.unbind(),
        T.unbind(),
        x.unbind(),
        P.unbind(),
        C.unbind(),
        L.unbind(),
        window.removeEventListener("resize", ot, !1),
        v.slides.removeEventListener("click", at, !1),
        v.slides.removeEventListener("transitionend", rt, !1),
        v.pauseOverlay.removeEventListener("click", ye, !1);
    }
    function G() {
      $(),
        Ye(),
        ie(),
        D.destroy(),
        T.destroy(),
        M.destroy(),
        N.destroy(),
        P.destroy(),
        C.destroy(),
        k.destroy(),
        E.destroy(),
        document.removeEventListener("fullscreenchange", lt),
        document.removeEventListener("webkitfullscreenchange", lt),
        document.removeEventListener("visibilitychange", st, !1),
        window.removeEventListener("message", it, !1),
        window.removeEventListener("load", le, !1),
        v.pauseOverlay && v.pauseOverlay.remove(),
        v.statusElement && v.statusElement.remove(),
        document.documentElement.classList.remove("reveal-full-page"),
        v.wrapper.classList.remove(
          "ready",
          "center",
          "has-horizontal-slides",
          "has-vertical-slides",
        ),
        v.wrapper.removeAttribute("data-transition-speed"),
        v.wrapper.removeAttribute("data-background-transition"),
        v.viewport.classList.remove("reveal-viewport"),
        v.viewport.style.removeProperty("--slide-width"),
        v.viewport.style.removeProperty("--slide-height"),
        v.slides.style.removeProperty("width"),
        v.slides.style.removeProperty("height"),
        v.slides.style.removeProperty("zoom"),
        v.slides.style.removeProperty("left"),
        v.slides.style.removeProperty("top"),
        v.slides.style.removeProperty("bottom"),
        v.slides.style.removeProperty("right"),
        v.slides.style.removeProperty("transform"),
        Array.from(v.wrapper.querySelectorAll(tf)).forEach(function (e) {
          e.style.removeProperty("display"),
            e.style.removeProperty("top"),
            e.removeAttribute("hidden"),
            e.removeAttribute("aria-hidden");
        });
    }
    function J(t, n, i) {
      e.addEventListener(t, n, i);
    }
    function Q(t, n, i) {
      e.removeEventListener(t, n, i);
    }
    function Z(e) {
      "string" == typeof e.layout && (f.layout = e.layout),
        "string" == typeof e.overview && (f.overview = e.overview),
        f.layout
          ? Eh(v.slides, f.layout + " " + f.overview)
          : Eh(v.slides, f.overview);
    }
    function ee(e) {
      var t = e.target,
        n = void 0 === t ? v.wrapper : t,
        i = e.type,
        r = e.data,
        a = e.bubbles,
        o = void 0 === a || a,
        s = document.createEvent("HTMLEvents", 1, 2);
      return (
        s.initEvent(i, o, !0),
        mh(s, r),
        n.dispatchEvent(s),
        n === v.wrapper && te(i),
        s
      );
    }
    function te(e, t) {
      if (l.postMessageEvents && window.parent !== window.self) {
        var n = { namespace: "reveal", eventName: e, state: We() };
        mh(n, t), window.parent.postMessage(JSON.stringify(n), "*");
      }
    }
    function ne() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : "a";
      Array.from(v.wrapper.querySelectorAll(e)).forEach(function (e) {
        /^(http|www)/gi.test(e.getAttribute("href")) &&
          e.addEventListener("click", ct, !1);
      });
    }
    function ie() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : "a";
      Array.from(v.wrapper.querySelectorAll(e)).forEach(function (e) {
        /^(http|www)/gi.test(e.getAttribute("href")) &&
          e.removeEventListener("click", ct, !1);
      });
    }
    function re(e) {
      se(),
        (v.overlay = document.createElement("div")),
        v.overlay.classList.add("overlay"),
        v.overlay.classList.add("overlay-preview"),
        v.wrapper.appendChild(v.overlay),
        (v.overlay.innerHTML =
          '<header>\n\t\t\t\t<a class="close" href="#"><span class="icon"></span></a>\n\t\t\t\t<a class="external" href="'
            .concat(
              e,
              '" target="_blank"><span class="icon"></span></a>\n\t\t\t</header>\n\t\t\t<div class="spinner"></div>\n\t\t\t<div class="viewport">\n\t\t\t\t<iframe src="',
            )
            .concat(
              e,
              '"></iframe>\n\t\t\t\t<small class="viewport-inner">\n\t\t\t\t\t<span class="x-frame-error">Unable to load iframe. This is likely due to the site\'s policy (x-frame-options).</span>\n\t\t\t\t</small>\n\t\t\t</div>',
            )),
        v.overlay.querySelector("iframe").addEventListener(
          "load",
          function (e) {
            v.overlay.classList.add("loaded");
          },
          !1,
        ),
        v.overlay.querySelector(".close").addEventListener(
          "click",
          function (e) {
            se(), e.preventDefault();
          },
          !1,
        ),
        v.overlay.querySelector(".external").addEventListener(
          "click",
          function (e) {
            se();
          },
          !1,
        );
    }
    function ae(e) {
      "boolean" == typeof e ? (e ? oe() : se()) : v.overlay ? se() : oe();
    }
    function oe() {
      if (l.help) {
        se(),
          (v.overlay = document.createElement("div")),
          v.overlay.classList.add("overlay"),
          v.overlay.classList.add("overlay-help"),
          v.wrapper.appendChild(v.overlay);
        var e = '<p class="title">Keyboard Shortcuts</p><br/>',
          t = x.getShortcuts(),
          n = x.getBindings();
        for (var i in ((e += "<table><th>KEY</th><th>ACTION</th>"), t))
          e += "<tr><td>".concat(i, "</td><td>").concat(t[i], "</td></tr>");
        for (var r in n)
          n[r].key &&
            n[r].description &&
            (e += "<tr><td>"
              .concat(n[r].key, "</td><td>")
              .concat(n[r].description, "</td></tr>"));
        (e += "</table>"),
          (v.overlay.innerHTML =
            '\n\t\t\t\t<header>\n\t\t\t\t\t<a class="close" href="#"><span class="icon"></span></a>\n\t\t\t\t</header>\n\t\t\t\t<div class="viewport">\n\t\t\t\t\t<div class="viewport-inner">'.concat(
              e,
              "</div>\n\t\t\t\t</div>\n\t\t\t",
            )),
          v.overlay.querySelector(".close").addEventListener(
            "click",
            function (e) {
              se(), e.preventDefault();
            },
            !1,
          );
      }
    }
    function se() {
      return (
        !!v.overlay &&
        (v.overlay.parentNode.removeChild(v.overlay), (v.overlay = null), !0)
      );
    }
    function le() {
      if (v.wrapper && !I.isPrintingPDF()) {
        if (!l.disableLayout) {
          Mh &&
            !l.embedded &&
            document.documentElement.style.setProperty(
              "--vh",
              0.01 * window.innerHeight + "px",
            );
          var e = ue(),
            t = h;
          ce(l.width, l.height),
            (v.slides.style.width = e.width + "px"),
            (v.slides.style.height = e.height + "px"),
            (h = Math.min(
              e.presentationWidth / e.width,
              e.presentationHeight / e.height,
            )),
            (h = Math.max(h, l.minScale)),
            1 === (h = Math.min(h, l.maxScale))
              ? ((v.slides.style.zoom = ""),
                (v.slides.style.left = ""),
                (v.slides.style.top = ""),
                (v.slides.style.bottom = ""),
                (v.slides.style.right = ""),
                Z({ layout: "" }))
              : ((v.slides.style.zoom = ""),
                (v.slides.style.left = "50%"),
                (v.slides.style.top = "50%"),
                (v.slides.style.bottom = "auto"),
                (v.slides.style.right = "auto"),
                Z({ layout: "translate(-50%, -50%) scale(" + h + ")" }));
          for (
            var n = Array.from(v.wrapper.querySelectorAll(tf)),
              i = 0,
              r = n.length;
            i < r;
            i++
          ) {
            var a = n[i];
            "none" !== a.style.display &&
              (l.center || a.classList.contains("center")
                ? a.classList.contains("stack")
                  ? (a.style.top = 0)
                  : (a.style.top =
                      Math.max((e.height - a.scrollHeight) / 2, 0) + "px")
                : (a.style.top = ""));
          }
          t !== h &&
            ee({ type: "resize", data: { oldScale: t, scale: h, size: e } });
        }
        v.viewport.style.setProperty("--slide-scale", h),
          C.update(),
          k.updateParallax(),
          R.isActive() && R.update();
      }
    }
    function ce(e, t) {
      yh(v.slides, "section > .stretch, section > .r-stretch").forEach(
        function (n) {
          var i = Lh(n, t);
          if (/(img|video)/gi.test(n.nodeName)) {
            var r = n.naturalWidth || n.videoWidth,
              a = n.naturalHeight || n.videoHeight,
              o = Math.min(e / r, i / a);
            (n.style.width = r * o + "px"), (n.style.height = a * o + "px");
          } else (n.style.width = e + "px"), (n.style.height = i + "px");
        },
      );
    }
    function ue(e, t) {
      var n = {
        width: l.width,
        height: l.height,
        presentationWidth: e || v.wrapper.offsetWidth,
        presentationHeight: t || v.wrapper.offsetHeight,
      };
      return (
        (n.presentationWidth -= n.presentationWidth * l.margin),
        (n.presentationHeight -= n.presentationHeight * l.margin),
        "string" == typeof n.width &&
          /%$/.test(n.width) &&
          (n.width = (parseInt(n.width, 10) / 100) * n.presentationWidth),
        "string" == typeof n.height &&
          /%$/.test(n.height) &&
          (n.height = (parseInt(n.height, 10) / 100) * n.presentationHeight),
        n
      );
    }
    function de(e, t) {
      "object" === Ei(e) &&
        "function" == typeof e.setAttribute &&
        e.setAttribute("data-previous-indexv", t || 0);
    }
    function he(e) {
      if (
        "object" === Ei(e) &&
        "function" == typeof e.setAttribute &&
        e.classList.contains("stack")
      ) {
        var t = e.hasAttribute("data-start-indexv")
          ? "data-start-indexv"
          : "data-previous-indexv";
        return parseInt(e.getAttribute(t) || 0, 10);
      }
      return 0;
    }
    function fe() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : a;
      return e && e.parentNode && !!e.parentNode.nodeName.match(/section/i);
    }
    function ve() {
      return !(!a || !fe(a)) && !a.nextElementSibling;
    }
    function pe() {
      return 0 === n && 0 === i;
    }
    function ge() {
      return (
        !!a &&
        !a.nextElementSibling &&
        (!fe(a) || !a.parentNode.nextElementSibling)
      );
    }
    function me() {
      if (l.pause) {
        var e = v.wrapper.classList.contains("paused");
        Ye(),
          v.wrapper.classList.add("paused"),
          !1 === e && ee({ type: "paused" });
      }
    }
    function ye() {
      var e = v.wrapper.classList.contains("paused");
      v.wrapper.classList.remove("paused"), Ke(), e && ee({ type: "resumed" });
    }
    function be(e) {
      "boolean" == typeof e ? (e ? me() : ye()) : we() ? ye() : me();
    }
    function we() {
      return v.wrapper.classList.contains("paused");
    }
    function Ee(e) {
      "boolean" == typeof e ? (e ? $e() : Xe()) : b ? $e() : Xe();
    }
    function Se() {
      return !(!g || b);
    }
    function ke(e, t, o, s) {
      if (
        !ee({
          type: "beforeslidechange",
          data: {
            indexh: void 0 === e ? n : e,
            indexv: void 0 === t ? i : t,
            origin: s,
          },
        }).defaultPrevented
      ) {
        r = a;
        var c = v.wrapper.querySelectorAll(nf);
        if (0 !== c.length) {
          void 0 !== t || R.isActive() || (t = he(c[e])),
            r &&
              r.parentNode &&
              r.parentNode.classList.contains("stack") &&
              de(r.parentNode, i);
          var u = d.concat();
          d.length = 0;
          var h = n || 0,
            f = i || 0;
          (n = Pe(nf, void 0 === e ? n : e)),
            (i = Pe(rf, void 0 === t ? i : t));
          var g = n !== h || i !== f;
          g || (r = null);
          var m = c[n],
            y = m.querySelectorAll("section");
          a = y[i] || m;
          var b = !1;
          g &&
            r &&
            a &&
            !R.isActive() &&
            (r.hasAttribute("data-auto-animate") &&
              a.hasAttribute("data-auto-animate") &&
              r.getAttribute("data-auto-animate-id") ===
                a.getAttribute("data-auto-animate-id") &&
              !(n > h || i > f ? a : r).hasAttribute(
                "data-auto-animate-restart",
              ) &&
              ((b = !0), v.slides.classList.add("disable-slide-transitions")),
            (p = "running")),
            Ce(),
            le(),
            R.isActive() && R.update(),
            void 0 !== o && A.goto(o),
            r &&
              r !== a &&
              (r.classList.remove("present"),
              r.setAttribute("aria-hidden", "true"),
              pe() &&
                setTimeout(function () {
                  Fe().forEach(function (e) {
                    de(e, 0);
                  });
                }, 0));
          e: for (var x = 0, N = d.length; x < N; x++) {
            for (var M = 0; M < u.length; M++)
              if (u[M] === d[x]) {
                u.splice(M, 1);
                continue e;
              }
            v.viewport.classList.add(d[x]), ee({ type: d[x] });
          }
          for (; u.length; ) v.viewport.classList.remove(u.pop());
          g &&
            ee({
              type: "slidechanged",
              data: {
                indexh: n,
                indexv: i,
                previousSlide: r,
                currentSlide: a,
                origin: s,
              },
            }),
            (!g && r) || (w.stopEmbeddedContent(r), w.startEmbeddedContent(a)),
            requestAnimationFrame(function () {
              B(q(a));
            }),
            C.update(),
            P.update(),
            D.update(),
            k.update(),
            k.updateParallax(),
            E.update(),
            A.update(),
            L.writeURL(),
            Ke(),
            b &&
              (setTimeout(function () {
                v.slides.classList.remove("disable-slide-transitions");
              }, 0),
              l.autoAnimate && S.run(r, a));
        }
      }
    }
    function Ae() {
      $(),
        X(),
        le(),
        (g = l.autoSlide),
        Ke(),
        k.create(),
        L.writeURL(),
        A.sortAll(),
        P.update(),
        C.update(),
        Ce(),
        D.update(),
        D.updateVisibility(),
        k.update(!0),
        E.update(),
        w.formatEmbeddedContent(),
        !1 === l.autoPlayMedia
          ? w.stopEmbeddedContent(a, { unloadIframes: !1 })
          : w.startEmbeddedContent(a),
        R.isActive() && R.layout();
    }
    function Re() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : a;
      k.sync(e), A.sync(e), w.load(e), k.update(), D.update();
    }
    function xe() {
      De().forEach(function (e) {
        yh(e, "section").forEach(function (e, t) {
          t > 0 &&
            (e.classList.remove("present"),
            e.classList.remove("past"),
            e.classList.add("future"),
            e.setAttribute("aria-hidden", "true"));
        });
      });
    }
    function Le() {
      var e =
        arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : De();
      e.forEach(function (t, n) {
        var i = e[Math.floor(Math.random() * e.length)];
        i.parentNode === t.parentNode && t.parentNode.insertBefore(t, i);
        var r = t.querySelectorAll("section");
        r.length && Le(r);
      });
    }
    function Pe(e, t) {
      var n = yh(v.wrapper, e),
        i = n.length,
        r = I.isPrintingPDF();
      if (i) {
        l.loop && (t %= i) < 0 && (t = i + t),
          (t = Math.max(Math.min(t, i - 1), 0));
        for (var a = 0; a < i; a++) {
          var o = n[a],
            s = l.rtl && !fe(o);
          o.classList.remove("past"),
            o.classList.remove("present"),
            o.classList.remove("future"),
            o.setAttribute("hidden", ""),
            o.setAttribute("aria-hidden", "true"),
            o.querySelector("section") && o.classList.add("stack"),
            r
              ? o.classList.add("present")
              : a < t
              ? (o.classList.add(s ? "future" : "past"),
                l.fragments &&
                  yh(o, ".fragment").forEach(function (e) {
                    e.classList.add("visible"),
                      e.classList.remove("current-fragment");
                  }))
              : a > t &&
                (o.classList.add(s ? "past" : "future"),
                l.fragments &&
                  yh(o, ".fragment.visible").forEach(function (e) {
                    e.classList.remove("visible", "current-fragment");
                  }));
        }
        var c = n[t],
          u = c.classList.contains("present");
        c.classList.add("present"),
          c.removeAttribute("hidden"),
          c.removeAttribute("aria-hidden"),
          u || ee({ target: c, type: "visible", bubbles: !1 });
        var h = c.getAttribute("data-state");
        h && (d = d.concat(h.split(" ")));
      } else t = 0;
      return t;
    }
    function Ce() {
      var e,
        t = De(),
        r = t.length;
      if (r && void 0 !== n) {
        var a = R.isActive() ? 10 : l.viewDistance;
        Mh && (a = R.isActive() ? 6 : l.mobileViewDistance),
          I.isPrintingPDF() && (a = Number.MAX_VALUE);
        for (var o = 0; o < r; o++) {
          var s = t[o],
            c = yh(s, "section"),
            u = c.length;
          if (
            ((e = Math.abs((n || 0) - o) || 0),
            l.loop && (e = Math.abs(((n || 0) - o) % (r - a)) || 0),
            e < a ? w.load(s) : w.unload(s),
            u)
          )
            for (var d = he(s), h = 0; h < u; h++) {
              var f = c[h];
              e + (o === (n || 0) ? Math.abs((i || 0) - h) : Math.abs(h - d)) <
              a
                ? w.load(f)
                : w.unload(f);
            }
        }
        He()
          ? v.wrapper.classList.add("has-vertical-slides")
          : v.wrapper.classList.remove("has-vertical-slides"),
          ze()
            ? v.wrapper.classList.add("has-horizontal-slides")
            : v.wrapper.classList.remove("has-horizontal-slides");
      }
    }
    function Ne() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.includeFragments,
        r = void 0 !== t && t,
        a = v.wrapper.querySelectorAll(nf),
        o = v.wrapper.querySelectorAll(rf),
        s = {
          left: n > 0,
          right: n < a.length - 1,
          up: i > 0,
          down: i < o.length - 1,
        };
      if (
        (l.loop &&
          (a.length > 1 && ((s.left = !0), (s.right = !0)),
          o.length > 1 && ((s.up = !0), (s.down = !0))),
        a.length > 1 &&
          "linear" === l.navigationMode &&
          ((s.right = s.right || s.down), (s.left = s.left || s.up)),
        !0 === r)
      ) {
        var c = A.availableRoutes();
        (s.left = s.left || c.prev),
          (s.up = s.up || c.prev),
          (s.down = s.down || c.next),
          (s.right = s.right || c.next);
      }
      if (l.rtl) {
        var u = s.left;
        (s.left = s.right), (s.right = u);
      }
      return s;
    }
    function Me() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : a,
        t = De(),
        n = 0;
      e: for (var i = 0; i < t.length; i++) {
        for (
          var r = t[i], o = r.querySelectorAll("section"), s = 0;
          s < o.length;
          s++
        ) {
          if (o[s] === e) break e;
          "uncounted" !== o[s].dataset.visibility && n++;
        }
        if (r === e) break;
        !1 === r.classList.contains("stack") &&
          "uncounted" !== r.dataset.visibility &&
          n++;
      }
      return n;
    }
    function Ie() {
      var e = _e(),
        t = Me();
      if (a) {
        var n = a.querySelectorAll(".fragment");
        if (n.length > 0) {
          t +=
            (a.querySelectorAll(".fragment.visible").length / n.length) * 0.9;
        }
      }
      return Math.min(t / (e - 1), 1);
    }
    function Te(e) {
      var t,
        r = n,
        o = i;
      if (e) {
        var s = fe(e),
          l = s ? e.parentNode : e,
          c = De();
        (r = Math.max(c.indexOf(l), 0)),
          (o = void 0),
          s && (o = Math.max(yh(e.parentNode, "section").indexOf(e), 0));
      }
      if (!e && a && a.querySelectorAll(".fragment").length > 0) {
        var u = a.querySelector(".current-fragment");
        t =
          u && u.hasAttribute("data-fragment-index")
            ? parseInt(u.getAttribute("data-fragment-index"), 10)
            : a.querySelectorAll(".fragment.visible").length - 1;
      }
      return { h: r, v: o, f: t };
    }
    function Oe() {
      return yh(
        v.wrapper,
        '.slides section:not(.stack):not([data-visibility="uncounted"])',
      );
    }
    function De() {
      return yh(v.wrapper, nf);
    }
    function je() {
      return yh(v.wrapper, ".slides>section>section");
    }
    function Fe() {
      return yh(v.wrapper, ".slides>section.stack");
    }
    function ze() {
      return De().length > 1;
    }
    function He() {
      return je().length > 1;
    }
    function Ue() {
      return Oe().map(function (e) {
        for (var t = {}, n = 0; n < e.attributes.length; n++) {
          var i = e.attributes[n];
          t[i.name] = i.value;
        }
        return t;
      });
    }
    function _e() {
      return Oe().length;
    }
    function Be(e, t) {
      var n = De()[e],
        i = n && n.querySelectorAll("section");
      return i && i.length && "number" == typeof t ? (i ? i[t] : void 0) : n;
    }
    function qe(e, t) {
      var n = "number" == typeof e ? Be(e, t) : e;
      if (n) return n.slideBackgroundElement;
    }
    function We() {
      var e = Te();
      return {
        indexh: e.h,
        indexv: e.v,
        indexf: e.f,
        paused: we(),
        overview: R.isActive(),
      };
    }
    function Ve(e) {
      if ("object" === Ei(e)) {
        ke(wh(e.indexh), wh(e.indexv), wh(e.indexf));
        var t = wh(e.paused),
          n = wh(e.overview);
        "boolean" == typeof t && t !== we() && be(t),
          "boolean" == typeof n && n !== R.isActive() && R.toggle(n);
      }
    }
    function Ke() {
      if ((Ye(), a && !1 !== l.autoSlide)) {
        var e = a.querySelector(".current-fragment");
        e || (e = a.querySelector(".fragment"));
        var t = e ? e.getAttribute("data-autoslide") : null,
          n = a.parentNode ? a.parentNode.getAttribute("data-autoslide") : null,
          i = a.getAttribute("data-autoslide");
        t
          ? (g = parseInt(t, 10))
          : i
          ? (g = parseInt(i, 10))
          : n
          ? (g = parseInt(n, 10))
          : ((g = l.autoSlide),
            0 === a.querySelectorAll(".fragment").length &&
              yh(a, "video, audio").forEach(function (e) {
                e.hasAttribute("data-autoplay") &&
                  g &&
                  (1e3 * e.duration) / e.playbackRate > g &&
                  (g = (1e3 * e.duration) / e.playbackRate + 1e3);
              })),
          !g ||
            b ||
            we() ||
            R.isActive() ||
            (ge() && !A.availableRoutes().next && !0 !== l.loop) ||
            ((m = setTimeout(function () {
              "function" == typeof l.autoSlideMethod
                ? l.autoSlideMethod()
                : tt(),
                Ke();
            }, g)),
            (y = Date.now())),
          o && o.setPlaying(-1 !== m);
      }
    }
    function Ye() {
      clearTimeout(m), (m = -1);
    }
    function Xe() {
      g &&
        !b &&
        ((b = !0),
        ee({ type: "autoslidepaused" }),
        clearTimeout(m),
        o && o.setPlaying(!1));
    }
    function $e() {
      g && b && ((b = !1), ee({ type: "autoslideresumed" }), Ke());
    }
    function Ge() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.skipFragments,
        r = void 0 !== t && t;
      (u.hasNavigatedHorizontally = !0),
        l.rtl
          ? (R.isActive() || r || !1 === A.next()) &&
            Ne().left &&
            ke(n + 1, "grid" === l.navigationMode ? i : void 0)
          : (R.isActive() || r || !1 === A.prev()) &&
            Ne().left &&
            ke(n - 1, "grid" === l.navigationMode ? i : void 0);
    }
    function Je() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.skipFragments,
        r = void 0 !== t && t;
      (u.hasNavigatedHorizontally = !0),
        l.rtl
          ? (R.isActive() || r || !1 === A.prev()) &&
            Ne().right &&
            ke(n - 1, "grid" === l.navigationMode ? i : void 0)
          : (R.isActive() || r || !1 === A.next()) &&
            Ne().right &&
            ke(n + 1, "grid" === l.navigationMode ? i : void 0);
    }
    function Qe() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.skipFragments,
        r = void 0 !== t && t;
      (R.isActive() || r || !1 === A.prev()) && Ne().up && ke(n, i - 1);
    }
    function Ze() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.skipFragments,
        r = void 0 !== t && t;
      (u.hasNavigatedVertically = !0),
        (R.isActive() || r || !1 === A.next()) && Ne().down && ke(n, i + 1);
    }
    function et() {
      var e,
        t = arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        i = t.skipFragments,
        r = void 0 !== i && i;
      if (r || !1 === A.prev())
        if (Ne().up) Qe({ skipFragments: r });
        else if (
          (e = l.rtl
            ? yh(v.wrapper, ".slides>section.future").pop()
            : yh(v.wrapper, ".slides>section.past").pop()) &&
          e.classList.contains("stack")
        ) {
          var a = e.querySelectorAll("section").length - 1 || void 0,
            o = n - 1;
          ke(o, a);
        } else Ge({ skipFragments: r });
    }
    function tt() {
      var e =
          arguments.length > 0 && void 0 !== arguments[0] ? arguments[0] : {},
        t = e.skipFragments,
        n = void 0 !== t && t;
      if (
        ((u.hasNavigatedHorizontally = !0),
        (u.hasNavigatedVertically = !0),
        n || !1 === A.next())
      ) {
        var i = Ne();
        i.down && i.right && l.loop && ve() && (i.down = !1),
          i.down
            ? Ze({ skipFragments: n })
            : l.rtl
            ? Ge({ skipFragments: n })
            : Je({ skipFragments: n });
      }
    }
    function nt(e) {
      l.autoSlideStoppable && Xe();
    }
    function it(e) {
      var t = e.data;
      if (
        "string" == typeof t &&
        "{" === t.charAt(0) &&
        "}" === t.charAt(t.length - 1) &&
        (t = JSON.parse(t)).method &&
        "function" == typeof s[t.method]
      )
        if (!1 === af.test(t.method)) {
          var n = s[t.method].apply(s, t.args);
          te("callback", { method: t.method, result: n });
        } else
          console.warn(
            'reveal.js: "' +
              t.method +
              '" is is blacklisted from the postMessage API',
          );
    }
    function rt(e) {
      "running" === p &&
        /section/gi.test(e.target.nodeName) &&
        ((p = "idle"),
        ee({
          type: "slidetransitionend",
          data: { indexh: n, indexv: i, previousSlide: r, currentSlide: a },
        }));
    }
    function at(e) {
      var t = kh(e.target, 'a[href^="#"]');
      if (t) {
        var n = t.getAttribute("href"),
          i = L.getIndicesFromHash(n);
        i && (s.slide(i.h, i.v, i.f), e.preventDefault());
      }
    }
    function ot(e) {
      le();
    }
    function st(e) {
      !1 === document.hidden &&
        document.activeElement !== document.body &&
        ("function" == typeof document.activeElement.blur &&
          document.activeElement.blur(),
        document.body.focus());
    }
    function lt(e) {
      (document.fullscreenElement || document.webkitFullscreenElement) ===
        v.wrapper &&
        (e.stopImmediatePropagation(),
        setTimeout(function () {
          s.layout(), s.focus.focus();
        }, 1));
    }
    function ct(e) {
      if (e.currentTarget && e.currentTarget.hasAttribute("href")) {
        var t = e.currentTarget.getAttribute("href");
        t && (re(t), e.preventDefault());
      }
    }
    function ut(e) {
      ge() && !1 === l.loop ? (ke(0, 0), $e()) : b ? $e() : Xe();
    }
    var dt = {
      VERSION: Uf,
      initialize: j,
      configure: Y,
      destroy: G,
      sync: Ae,
      syncSlide: Re,
      syncFragments: A.sync.bind(A),
      slide: ke,
      left: Ge,
      right: Je,
      up: Qe,
      down: Ze,
      prev: et,
      next: tt,
      navigateLeft: Ge,
      navigateRight: Je,
      navigateUp: Qe,
      navigateDown: Ze,
      navigatePrev: et,
      navigateNext: tt,
      navigateFragment: A.goto.bind(A),
      prevFragment: A.prev.bind(A),
      nextFragment: A.next.bind(A),
      on: J,
      off: Q,
      addEventListener: J,
      removeEventListener: Q,
      layout: le,
      shuffle: Le,
      availableRoutes: Ne,
      availableFragments: A.availableRoutes.bind(A),
      toggleHelp: ae,
      toggleOverview: R.toggle.bind(R),
      togglePause: be,
      toggleAutoSlide: Ee,
      isFirstSlide: pe,
      isLastSlide: ge,
      isLastVerticalSlide: ve,
      isVerticalSlide: fe,
      isPaused: we,
      isAutoSliding: Se,
      isSpeakerNotes: D.isSpeakerNotesWindow.bind(D),
      isOverview: R.isActive.bind(R),
      isFocused: T.isFocused.bind(T),
      isPrintingPDF: I.isPrintingPDF.bind(I),
      isReady: function () {
        return c;
      },
      loadSlide: w.load.bind(w),
      unloadSlide: w.unload.bind(w),
      showPreview: re,
      hidePreview: se,
      addEventListeners: X,
      removeEventListeners: $,
      dispatchEvent: ee,
      getState: We,
      setState: Ve,
      getProgress: Ie,
      getIndices: Te,
      getSlidesAttributes: Ue,
      getSlidePastCount: Me,
      getTotalSlides: _e,
      getSlide: Be,
      getPreviousSlide: function () {
        return r;
      },
      getCurrentSlide: function () {
        return a;
      },
      getSlideBackground: qe,
      getSlideNotes: D.getSlideNotes.bind(D),
      getSlides: Oe,
      getHorizontalSlides: De,
      getVerticalSlides: je,
      hasHorizontalSlides: ze,
      hasVerticalSlides: He,
      hasNavigatedHorizontally: function () {
        return u.hasNavigatedHorizontally;
      },
      hasNavigatedVertically: function () {
        return u.hasNavigatedVertically;
      },
      addKeyBinding: x.addKeyBinding.bind(x),
      removeKeyBinding: x.removeKeyBinding.bind(x),
      triggerKey: x.triggerKey.bind(x),
      registerKeyboardShortcut: x.registerKeyboardShortcut.bind(x),
      getComputedSlideSize: ue,
      getScale: function () {
        return h;
      },
      getConfig: function () {
        return l;
      },
      getQueryHash: xh,
      getSlidePath: L.getHash.bind(L),
      getRevealElement: function () {
        return e;
      },
      getSlidesElement: function () {
        return v.slides;
      },
      getViewportElement: function () {
        return v.viewport;
      },
      getBackgroundsElement: function () {
        return k.element;
      },
      registerPlugin: M.registerPlugin.bind(M),
      hasPlugin: M.hasPlugin.bind(M),
      getPlugin: M.getPlugin.bind(M),
      getPlugins: M.getRegisteredPlugins.bind(M),
    };
    return (
      mh(
        s,
        wi(
          wi({}, dt),
          {},
          {
            announceStatus: B,
            getStatusText: q,
            print: I,
            focus: T,
            progress: C,
            controls: P,
            location: L,
            overview: R,
            fragments: A,
            slideContent: w,
            slideNumber: E,
            onUserInput: nt,
            closeOverlay: se,
            updateSlidesVisibility: Ce,
            layoutSlideContents: ce,
            transformSlides: Z,
            cueAutoSlide: Ke,
            cancelAutoSlide: Ye,
          },
        ),
      ),
      dt
    );
  }
  var Bf = _f,
    qf = [];
  return (
    (Bf.initialize = function (e) {
      return (
        Object.assign(Bf, new _f(document.querySelector(".reveal"), e)),
        qf.map(function (e) {
          return e(Bf);
        }),
        Bf.initialize()
      );
    }),
    [
      "configure",
      "on",
      "off",
      "addEventListener",
      "removeEventListener",
      "registerPlugin",
    ].forEach(function (e) {
      Bf[e] = function () {
        for (var t = arguments.length, n = new Array(t), i = 0; i < t; i++)
          n[i] = arguments[i];
        qf.push(function (t) {
          var i;
          return (i = t[e]).call.apply(i, [null].concat(n));
        });
      };
    }),
    (Bf.isReady = function () {
      return !1;
    }),
    (Bf.VERSION = Uf),
    Bf
  );
});
//# sourceMappingURL=reveal.js.map
