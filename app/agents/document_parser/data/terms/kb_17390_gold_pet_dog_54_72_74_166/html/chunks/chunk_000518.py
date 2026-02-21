from langchain_core.documents import Document

chunk = Document(
    page_content=("어 풀 이 신의료기술평가위원회</td></tr></tbody></table><br><p id='255' "
 "data-category='paragraph' style='font-size:16px'>의료법 제54조(신의료기술평가위원회의 설치 등)에 "
 "의거 설치된 위원회로서 신</p><br><h1 id='256' style='font-size:16px'>의료기술에 관한 최고의 심의기구를 "
 "말합니다.</h1><br><table id='257' "
 "style='font-size:16px'><thead></thead><tbody><tr><td"),
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
