from langchain_core.documents import Document

chunk = Document(
    page_content=('① 지정대리청구인은 제6조(보험금의 청구)에 정한 구비서류 및 제1조(적용대상)의 보험 수익자가 보험금을 직접 청구할 수 없는 특별한 '
 '사정이 있음을 증명하는 서류를 제출 하고 회사의 승낙을 얻어 제1조(적용대상)의 보험수익자의 대리인으로서 보험금(사망 보험금 제외)을 '
 '청구하고 수령할 수 있습니다. 다만, 2인의 청구대리인이 지정된 경우 에는 그 중 대표대리인이 보험금을 청구하고 수령할 수 있으며, '
 '대표대리인이 사망 등 의 사유로 보험금 청구가 불가능한 경우에는 대표가 아닌 청구대리인도 보험금을 청 구하고 수령할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000638',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
