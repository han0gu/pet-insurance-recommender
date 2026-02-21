from langchain_core.documents import Document

chunk = Document(
    page_content=('. 손해배상책임의 전부 또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거<br>나 소송, 중재 또는 조정을 제기하거나 신청하고자 '
 "할 경우에는 미리 회사의</p><br><p id='203' data-category='paragraph' "
 "style='font-size:16px'>동의를 받는 일<br>\uf000 계약자 또는 피보험자가 정당한 이유없이 제1항의 의무를 "
 "이행하지 않았을 때에</p><br><p id='204' data-category='list' style='font-size:16px'>는 "
 '제3조(보상하는 손해)에 의한'),
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
 'indexing': {'chunk_id': 'chunk_001183',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
