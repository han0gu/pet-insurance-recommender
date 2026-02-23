from langchain_core.documents import Document

chunk = Document(
    page_content=('말합니다.<br>\uf000 제1항의 수술에서 보건복지부 산하 신의료기술평가위원회(향후 제도 변경 시에는<br>동 위원회와 동일한 기능을 '
 '수행하는 기관) 또는 이에 준하는 기관으로부터 안전<br>성과 치료효과를 인정받은 최신 수술기법도 포함됩니다.</p><br><table '
 "id='111' style='font-size:14px'><thead></thead><tbody><tr><td>용 어 "
 '풀</td><td>이 신의료기술평가위원회</td></tr><tr><td colspan="2">의료법 제54조(신의료기술평가위원회의 설치 '
 '등)에 의거'),
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
