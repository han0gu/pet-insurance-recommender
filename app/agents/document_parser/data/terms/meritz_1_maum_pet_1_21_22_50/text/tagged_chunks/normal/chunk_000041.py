from langchain_core.documents import Document

chunk = Document(
    page_content=('지급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할 때에는 아래에 따라 보\n'
 '험금을 지급합니다.피보험자가 부담한 총 비용금액×이 계약의 지급보험금다른 계약이 없는 것으로 하여 각각계산한 지급보험금의 합계액② '
 '피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 따른\n'
 '지급보험금 결정에는 영향을 미치지 않습니다.제11조(보험금 받는 방법의 변경)① 계약자(보험금 지급사유 발생 후에는 보험수익자)는 회사의 '
 '사업방법서에서 정한 바에'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
