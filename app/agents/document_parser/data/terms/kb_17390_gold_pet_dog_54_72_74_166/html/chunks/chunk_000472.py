from langchain_core.documents import Document

chunk = Document(
    page_content=('어 풀 이 신의료기술평가위원회 의료법 제54조(신의료기술평가위원회의 설치 등)에 의거 설치된 위원회로서 '
 "신</td></tr></tbody></table><br><h1 id='180' style='font-size:14px'>의료기술에 관한 "
 "최고의 심의기구를 말합니다.</h1><br><p id='181' data-category='paragraph' "
 "style='font-size:14px'>\uf000</p><br><p id='182' data-category='list' "
 "style='font-size:14px'>제1항의 수술에서 아래에"),
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
