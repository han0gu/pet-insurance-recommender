from langchain_core.documents import Document

chunk = Document(
    page_content=('1,000<br>만원)<br>→ 계약B보험회사 : 500만원 지급 = 1,000만원 × 1,000만원 / (1,000만원 + '
 "1,000<br>만원)</p><br><p id='65' data-category='paragraph' "
 "style='font-size:14px'>② 이 특별약관이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 "
 "의무보험에서<br>보상되는 금액(피보험자가 가입을 하지 않은 경우에는 보상될 것으로 추정되는 금액)을</p><footer id='66' "
 "style='font-size:14px'>- 25"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000239',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
