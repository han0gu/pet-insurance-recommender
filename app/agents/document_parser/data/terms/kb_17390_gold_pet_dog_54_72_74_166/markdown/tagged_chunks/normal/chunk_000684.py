from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리를 지키거나 행사\n'
 '- 하기 위한 필요한 조치를 취하는 일\n'
 '- 3. 손해배상책임의 전부 또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거\n'
 '- 나 소송, 중재 또는 조정을 제기하거나 신청하고자 할 경우에는 미리 회사의\n'
 '동의를 받는 일\n'
 '\uf000 계약자 또는 피보험자가 정당한 이유없이 제1항의 의무를 이행하지 않았을 때에- 는 제3조(보상하는 손해)에 의한 손해에서 '
 '다음의 금액을 뺍니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000684',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
