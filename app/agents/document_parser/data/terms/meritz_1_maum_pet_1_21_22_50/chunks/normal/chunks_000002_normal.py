from langchain_core.documents import Document

chunk = Document(
    page_content=('. 마. 피보험자: 반려동물의 소유와 관련하여 보험사고로 손해를 입은 사람(법인인 경우 에는 그 이사 또는 법인의 업무를 집행하는 그 '
 '밖의 기관)을 말하며, 보험증권에 기재된 반려동물의 소유자 및 그 가족에 한합니다. 바. 반려동물 : 보험증권에 기재된 반려동물을 '
 '말하며, 이 계약에서 가입 가능한 반려동 물은 대한민국 내에서 피보험자와 거주를 함께하고 있는 개(犬) 또는 고양이(猫)를 말합니다. '
 '다만 아래에 기재된 개(犬) 또는 고양이(猫)는 이 보험의 가입 대상이 아 닙니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 1},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000002',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
