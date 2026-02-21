from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>병</p><table id='0' "
 "style='font-size:14px'><thead></thead><tbody><tr><td></td></tr><tr><td>용 어 풀 "
 '이 ∙ 자기공명영상(MRI) : 강한 자기장 내에서 인체에 고주파를 전사해서 반향 되 는 전자기파를 측정하는 영상진단법 ∙ '
 '컴퓨터단층촬영(CT) : x선을 투과시켜 그 흡수차이를 컴퓨터로 재구성하여</td></tr></tbody></table><br><p '
 "id='1' data-category='list'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
