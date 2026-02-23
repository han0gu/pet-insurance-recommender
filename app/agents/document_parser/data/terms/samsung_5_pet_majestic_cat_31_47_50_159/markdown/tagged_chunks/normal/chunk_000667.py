from langchain_core.documents import Document

chunk = Document(
    page_content=('- 「보험계약」이라 합니다)을 체결할 때 보험계약자(이하「계약자」라 합니다)의 청약\n'
 '- 과 보험회사(이하「회사」라 합니다)의 승낙으로 보험계약에 부가하여 이루어 집니다.\n'
 '- ② 이 특별약관에 대한 회사의 보장개시일(책임개시일)은 보험계약「제1회 보험료 및 회\n'
 '- 사의 보장개시」에서 정한 보장개시일(책임개시일)과 동일합니다.\n'
 '- ③ 이 특별약관은 피보험자가 이륜자동차를 소유, 사용(직업, 직무 또는 동호회 활동과\n'
 '- 출퇴근용도 등으로 주로 사용하는 경우에 한하며 일회적인 사용은 제외), 관리하는 경'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000667',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
