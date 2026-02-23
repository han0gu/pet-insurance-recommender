from langchain_core.documents import Document

chunk = Document(
    page_content=('id=\'36\' data-category=\'paragraph\' style=\'font-size:16px\'>Cast)치료"라 함은 '
 "【별표8】(부목치료 대상 분류</p><br><p id='37' data-category='list' "
 'style=\'font-size:16px\'>표)에서 정한 부목치료 대상 "수가코드"를 말하며, 국민건강보험법에서 정한 요<br>양급여 '
 '또는 의료급여법에서 정한 의료급여의 절차를 걸쳐 급여항목이 발생한 경<br>우를 말합니다.<br>\uf000 제1항의 부목치료는 '
 '"의사"에 의하여 부목치료가 필요하다고 인정된 경우로서'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000561',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
