from langchain_core.documents import Document

chunk = Document(
    page_content=('- 합니다.\n'
 '제3조(부목(Splint Cast)치료의 정의)# \uf000 이 특별약관에서 "부목(SplintCast)치료"라 함은 '
 '【별표8】(부목치료 대상 분류- 표)에서 정한 부목치료 대상 "수가코드"를 말하며, 국민건강보험법에서 정한 요\n'
 '- 양급여 또는 의료급여법에서 정한 의료급여의 절차를 걸쳐 급여항목이 발생한 경\n'
 '- 우를 말합니다.\n'
 '- \uf000 제1항의 부목치료는 "의사"에 의하여 부목치료가 필요하다고 인정된 경우로서 "\n'
 '- 의사"의 관리하에 의료법 제3조(의료기관) 제2항에서 규정한 국내의 병원 및 의'),
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
