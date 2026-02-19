from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중 에 제3항에서 정한 보장개시일(책임개시일) 이후에 '
 '보험증권에 기재된 반려묘가 보험 기간 중에 사망한 경우 보험증권에 기재된 보험가입금액을 보험수익자에게 보상하여 드립니다. ② 제1항의 '
 '사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경우 동 물병원에서 발급한 소견서를 제출하여야 합니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000683',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
