from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 신경계․정신행동 장해의 경우 ①<br>개호(장해로 혼자서 활동이 어려운 사람을 곁에서 돌<br>보는 것) 여부 ② 객관적 이유 '
 "및 개호의 내용을 추가<br>로 기재하여야 한다.</p><h1 id='1' style='font-size:20px'>\uf000 "
 "장해분류별 판정기준</h1><h1 id='2' style='font-size:20px'>1. 눈의 장해</h1><h1 id='3' "
 "style='font-size:20px'>가"),
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
