from langchain_core.documents import Document

chunk = Document(
    page_content=('① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 사항을 이행하여야 합니다.\n'
 '1. 손해의 방지 또는 경감을 위하여 노력하는 일(피해자에 대한 응급처치, 긴급호송 또 는 그 밖의 긴급조치를 포함합니다) 2. '
 '제3자로부터 손해의 배상을 받을 수 있는 경우에는 그 권리의 보전 또는 행사를 위 한 필요한 조치를 취하는 일 3. 손해배상책임의 전부 '
 '또는 일부에 관하여 지급(변제), 승인 또는 화해를 하거나 소 송, 중재 또는 조정을 제기하거나 신청하고자 할 경우에는 미리 회사의 '
 '동의를 받 는 일'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000160',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
