from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금 2. 계약자 또는 피보험자가 지출한 아래의 비용\n'
 '가. 피보험자가 제11조(손해방지의무)의 제1항 제1호의 손해의 방지 또는 경감을 위 하여 지출한 필요 또는 유익하였던 비용 나. '
 '피보험자가 제11조(손해방지의무)의 제1항 제2호의 제3자로부터 손해의 배상을 받을 수 있는 그 권리를 지키거나 행사하기 위하여 지출한 '
 '필요 또는 유익하였 던 비용'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000141',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
