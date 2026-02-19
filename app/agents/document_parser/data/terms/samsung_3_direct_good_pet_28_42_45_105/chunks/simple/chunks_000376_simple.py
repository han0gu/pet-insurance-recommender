from langchain_core.documents import Document

chunk = Document(
    page_content=('고용된 수의사는 해당 농장, 동물원 또는 수족관의 동물에게 투여할 목적으로 처방대상 동\n'
 '물용 의약품에 대한 처방전을 발급할 수 있다. 이 경우 상시고용된 수의사의 범위, 신고방\n'
 '법, 처방전 발급 및 보존 방법, 진료부 작성 및 보고, 교육, 준수사항 등 그 밖에 필요한 사 항은 농림축산식품부령으로 정한다.\n'
 '제 9조 (보험금의 지급절차)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 70},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000376',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
