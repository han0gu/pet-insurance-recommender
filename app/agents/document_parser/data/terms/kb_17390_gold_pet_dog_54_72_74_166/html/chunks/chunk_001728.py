from langchain_core.documents import Document

chunk = Document(
    page_content=(". 중수골, 중족골, 지골</td><td>N0976</td></tr></tbody></table><table id='58' "
 "style='font-size:16px'><thead></thead><tbody><tr><td></td><td "
 'colspan="2">별표8 부목치료 대상 분류표</td></tr><tr><td colspan="3">약관에서 규정하는 부목(Splint '
 'Cast)치료로 분류되는 의료행위는 "건강보험 행위 급 여 ․비급여 목록 및 급여 상대가치점수" 제2부(행위 급여목록․상대가치점수 및 '
 '산정지침) 의 제9장(처치 및'),
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
