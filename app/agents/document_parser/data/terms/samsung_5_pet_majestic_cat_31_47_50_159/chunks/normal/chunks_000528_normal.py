from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[공제계약]\n'
 '유사보험으로서 공제 사업을 실시하는 경영주체와 공제 계약자 사이에 체결되는 계약을 말합니다. 우체국, 신협, 새마을금고 등이 공제계약을 '
 '취급합니다.\n'
 '6. 중요한 사항: 계약 전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의 청약을 거절하거나 보험가입금액 한도 제한, 일부 '
 '보장 제외, 보험금 삭감, 보험료 할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합 니다.\n'
 '③ 지급금과 이자율 관련 용어'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 96},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 256,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
