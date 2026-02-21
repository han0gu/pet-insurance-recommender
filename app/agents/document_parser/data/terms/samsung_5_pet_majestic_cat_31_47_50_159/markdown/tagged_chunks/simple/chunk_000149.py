from langchain_core.documents import Document

chunk = Document(
    page_content=('당 "1억원까지" 예금자 보호가 됩니다. 이와 별도로 본 회사 보호상품의 사고보험금을 합산한 금액- 46 -이 1인당 "1억원까지" '
 '보호됩니다. 다만, 보험계약자 및 보험료 납부자가 법인인 보험계약의 경우\n'
 '에는 보호되지 않습니다.- 47 -※ 약관에서 인용된 법·규정은「별표 및 참고」의 「약관에서 인용된 법·규정」에서\n'
 '확인할 수 있습니다.특별약관 일반사항제1관 목적 및 용어의 정의# 제 1조 (목적)이 특별약관은 보험계약자(이하 " 계약자"라 합니다)와 '
 '보험회사(이하 " 회사"라 합니다) 사'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000149',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
