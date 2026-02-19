from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약을 체결할 때 계약에서 정한 피보험자 및 반려동물의 나이에 미달되었거나 초과되 었을 경우. 다만, 회사가 나이의 착오를 '
 '발견하였을 때 이미 계약나이에 도달한 경우에 는 유효한 계약으로 봅니다.\n'
 '제23조(계약내용의 변경 등)\n'
 '① 계약자는 회사의 승낙을 얻어 다음의 사항을 변경할 수 있습니다. 이 경우 승낙을 서면 등으로 알리거나 보험증권의 뒷면에 기재하여 '
 '드립니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 14},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000086',
              'chunk_char_len': 209,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
