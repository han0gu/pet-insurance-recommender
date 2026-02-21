from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또는 그 밖의 긴급조치를 포함합니다)\n'
 '- 2. 제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리를 지키거나 행사하기\n'
 '- 위한 필요한 조치를 취하는 일\n'
 '- 3. 손해배상책임의 전부 또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소\n'
 '- 송, 중재 또는 조정을 제기하거나 신청하고자 할 경우에는 미리 회사의 동의를 받\n'
 '- 는 일\n'
 '② 계약자 또는 피보험자가 정당한 이유 없이 제1항의 의무를 이행하지 않았을 때에는'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000651',
              'chunk_char_len': 241,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
