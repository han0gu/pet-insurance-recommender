from langchain_core.documents import Document

chunk = Document(
    page_content=('액에 대한 손해배상청구권을 가집니다.6. 중요한 사항 : 계약 전 알릴 의무와 관련하여 회사가 그 사실을 알았더라면 계약의청약을 '
 '거절하거나 보험가입금액 한도 제한, 일부 보장 제외, 보험금 삭감, 보험료\n'
 '할증과 같이 조건부로 승낙하는 등 계약 승낙에 영향을 미칠 수 있는 사항을 말합\n'
 '니다.# 제 3조 (보상하는 손해)- ① 회사는 대한민국 내에서 보험증권에 기재된 이 특별약관의 보험기간(이하「보험기간\n'
 '- 」이라 합니다) 중에 보험증권에 기재된 피보험자의 반려견의 행위에 기인하는 우연'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000637',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
