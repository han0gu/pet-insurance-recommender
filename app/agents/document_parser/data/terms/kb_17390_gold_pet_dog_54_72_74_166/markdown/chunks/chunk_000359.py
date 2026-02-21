from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다. 제3자는 의료법 제3조(의료기관)에 규정한 종합병원 소속 전문의 중에 정하\n'
 '며, 보험금 지급사유 판정에 드는 의료비용은 회사가 전액 부담합니다.# 제3조("창상봉합술(급여)"의 정의)# \uf000 이 특별약관에 '
 '있어서"창상봉합술(급여)"이라 함은 상해의 직접결과로써, "창상- 봉합술" 치료를 받은 경우를 말합니다. "창상봉합술"이라 함은 '
 '【별표9】(창상봉\n'
 '- 합술(안면/경부) 대상 수가코드)에서 정한 창상봉합술 대상 "수가코드"에 해당하\n'
 '- 는 경우를 말하며 해당 산정 기준일자는 치료개시일(해당 상병의 진료를 위하여 최'),
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
