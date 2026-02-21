from langchain_core.documents import Document

chunk = Document(
    page_content=('말한다.<br>4) 다만, 영구히 고정된 증상은 아니지만 치료종결후 한시<br>적으로 나타나는 장해에 대하여는 그 기간이 5년 '
 '이상<br>인 경우 해당장해 지급률의 20%를 장해지급률로 한다.<br>5) 위 4)에 따라 장해지급률이 결정되었으나 그 이후 '
 '보<br>장받을 수 있는 기간(계약의 효력이 없어진 경우에는<br>보험기간이 10년 이상인 계약은 상해 발생일 또는 질<br>병의 '
 '진단확정일부터 2년 이내로 하고, 보험기간이 10<br>년 미만인 계약은 상해 발생일 또는 질병의 진단확정<br>일부터 1년 이내)에 '
 '장해상태가 더'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
