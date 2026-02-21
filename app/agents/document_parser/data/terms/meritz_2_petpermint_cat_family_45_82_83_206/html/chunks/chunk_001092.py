from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, 질병발생 또는 상해를<br>입은 후 의식상실이 1개월 이상 지속된 경우에<br>는 질병발생 또는 상해를 입은 후 12개월이 '
 "지난<br>후에 판정할 수 있다.</p><br><p id='41' data-category='list' "
 "style='font-size:20px'>나) 정신행동장해는 장해판정 직전 1년 이상 충분한 정<br>신건강의학과의 전문적 치료를 받은 "
 '후 치료에도<br>불구하고 장해가 고착되었을 때 판정하여야 하며,<br>그렇지 않은 경우에는 그로써 고정되거나 중하게<br>된 장해에 '
 '대해서는 인정하지'),
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
