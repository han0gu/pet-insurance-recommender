from langchain_core.documents import Document

chunk = Document(
    page_content=("기타 위난이 종료한 후 1년간 분명하지 않은 때에도</p><br><p id='16' data-category='paragraph' "
 "style='font-size:14px'>제1항과 같습니다.</p><br><p id='17' "
 'data-category=\'paragraph\' style=\'font-size:14px\'>\uf000 "호스피스·완화의료 및 '
 '임종과정에 있는 환자의 연명의료 결정에 관한 법률"에</p><br><p id=\'18\' data-category=\'list\' '
 "style='font-size:14px'>따른 연명의료중단등결정 및 그"),
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
