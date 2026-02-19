from langchain_core.documents import Document

chunk = Document(
    page_content=('제7조(손해방지의무)\n'
 '① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 사항을 이행하여야 합니다.\n'
 '1. 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급호송 또는 그 밖의 긴급조치를 포함합니다) 2. '
 '제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리를 지키거나 행사하기 위한 필요한 조치를 취하는 일 3. 손해배상책임의 전부 '
 '또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소송, 중재 또는 조정을 제기하거나 신청하고자 할 경우에는 미리 회사의 동의를 '
 '받는 일'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
