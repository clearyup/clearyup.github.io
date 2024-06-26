# ACID

# 数据库的ACID原则
## 事务

>事务是一系列对系统中数据进行访问或更新的操作所组成的一个程序执行逻辑单元(Unit).为应用层服务的,而不是数据库系统本身的需要,`事务用来确保不论发生任何情况,数据始终处于一个合理的状态`


> 事务一般分为三个状态,Active,Commit,Failed

![w](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210422210036.png)

>完整的事务进一步放大看，事物内部还有部分提交这个中间状态，其对外是不可见的

![2](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210422210119.png)
## A(Atomicity)原子性

原子性,是指全有或者全无原则,事务的所有操作要么全部提交成功,要么全部失败回滚,不可能出现部分失败的情况,因为事务在执行的时候出问题,数据的错误是未知的,将它回滚到执行前的状态便于进行处理

>比如`A`转账给`B` ,这里`A`的账户余额要减少,`B`的账户余额要增加,增加和减少的操作要么都成功,要么都失败,不可能出现成功一个失败一个的情况

>刚说的是`操作`,为什么呢?因为这里如果是`A`转账给`B` `500元`,`A`账户减少	`500元`,`B`账户增加`300元`,这是符合原子性的!但是`A`减少的金额和`B`增加的金额并不相等,逻辑上有错误

为此数据库提供另一个原则`一致性`解决这个问题




## C(Consistency)一致性
一致性,是指事务在提交和回滚的变换过程中,数据保持一致性和正确性.

回到刚刚的问题
>`A`转账给`B` `500元`,`A`账户减少 `500元`,`B`账户增加`300元`,符合原子性!

>一致性不允许数据不一致,因此出现这种情况会回滚到转账前的状态

但是又会有这样的情况

>`A`和`C`同时向`B`转账`100元`,`B账户`原有余额为`300元`,事务执行之后`B`的账户应该有`500元`,但是`A`和`C`的事务同时读取`B`的账户余额,都执行`+100`,然后都得到`400`,都写入`400`,最后`B`的账户余额是`400元`

这一过程中,原子性符合,一致性也符合,但是数据还是错误,对此我们还需要另一个特性`隔离性`来约束它
## I(Isolation)隔离性
隔离性,一个事务所做的修改在最终提交前,对其他事务是不可见的.可以理解为在排队,前一个事务没有执行完,下一个事务不能操作前一个事务操作的数据.
>对于刚刚的问题虽然`A`和`B`同时是转账的,但是隔离性会在`A`转账成功之后再允许`C`转账,这样数据就不会得到`400`,而是`500`

满足原子性,一致性,隔离性,数据基本在事务执行前后可以保持一致正确,可要是在事务执行的过程中数据库突然崩溃,服务器突然断电,存储的数据会不会发生改变?这里引入最后一个特性`持久性`

## D(Durability)持久性
持久性,事务一旦提交,所做的修改就是永久性的.即使发生系统崩溃或者机器宕机等故障,只要数据库可以重新启动,就可以根据事务日志对`未持久化`的数据进行重新操作
>许多数据库通过引入**预写式日志**（Write-ahead logging，缩写 WAL）机制，来保证事务持久性和数据完整性，同时又很大程度上避免了基于事务直接刷新数据的频繁IO对性能的影响。

> 在使用WAL的系统中，所有的修改都先被写入到日志中，然后再被应用到系统状态中。假设一个程序在执行某些操作的过程中机器掉电了。在重新启动时，程序可能需要知道当时执行的操作是成功了还是部分成功或者是失败了。如果使用了WAL，程序就可以检查log文件，并对突然掉电时计划执行的操作内容跟实际上执行的操作内容进行比较。在这个比较的基础上，程序就可以决定是撤销已做的操作还是继续完成已做的操作，或者是保持原样。

## 总结
![2](https://cdn.jsdelivr.net/gh/clearyup/picgo/img/20210422215203.png)
