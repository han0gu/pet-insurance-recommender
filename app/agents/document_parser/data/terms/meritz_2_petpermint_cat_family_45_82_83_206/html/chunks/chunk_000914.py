from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 각 신체부위별 판정기준에서 별도<br>로 정한 경우에는 그 기준에 따른다.<br>3) 하나의 장해가 다른 장해와 통상 파생하는 '
 '관계에 있<br>는 경우에는 그중 높은 지급률만을 적용하며, 하나의<br>장해로 둘 이상의 파생장해가 발생하는 경우 각 파생<br>장해의 '
 '지급률을 합산한 지급률과 최초 장해의 지급률<br>을 비교하여 그 중 높은 지급률을 적용한다.<br>4) 의학적으로 뇌사판정을 받고 '
 '호흡기능과 심장박동기<br>능을 상실하여 인공심박동기 등 장치에 의존하여 생명<br>을 연장하고 있는 뇌사상태는 장해의 판정대상에'),
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
