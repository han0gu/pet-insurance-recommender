from langchain_core.documents import Document

chunk = Document(
    page_content=('. 손해배상책임의 전부 또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소<br>송, 중재 또는 조정을 제기하거나 신청하고자 '
 "할 경우에는 미리 회사의 동의를 받<br>는 일</p><br><p id='72' data-category='paragraph' "
 "style='font-size:14px'>② 계약자 또는 피보험자가 정당한 이유 없이 제1항의 의무를 이행하지 않았을 때에는 "
 "제<br>3조(보상하는 손해)에 의한 손해에서 다음의 금액을 뺍니다.</p><br><p id='73' "
 "data-category='list'"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000243',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
