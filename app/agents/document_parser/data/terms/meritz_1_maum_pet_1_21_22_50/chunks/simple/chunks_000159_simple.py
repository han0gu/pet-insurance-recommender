from langchain_core.documents import Document

chunk = Document(
    page_content=('차감한 금액을 손해액으로 간주하여 제1항에 의한 보상할 금액을 결정합니다.\n'
 '③ 피보험자가 다른 계약에 대하여 보험금 청구를 포기한 경우에도 회사의 제1항에 따른 지급보험금 결정에는 영향을 미치지 않습니다.\n'
 '제11조(손해방지의무)\n'
 '① 보험사고가 생긴 때에는 계약자 또는 피보험자는 아래의 사항을 이행하여야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 26},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000159',
              'chunk_char_len': 174,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
