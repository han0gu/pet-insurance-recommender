from langchain_core.documents import Document

chunk = Document(
    page_content=('익자의 책임있는 사유로 보험금 지급사유의 조사 및 확인이 지연되는 경우\n'
 '6. 제4조(보험금 지급에 관한 세부규정)에 따라 보험금 지급사유에 대해 제3자의 의견\n'
 '에 따르기로 한 경우- \n'
 '③ 제2항에 의하여 추가적인 조사가 이루어지는 경우, 회사는 보험수익자의 청구에 따라\n'
 '회사가 추정하는 보험금의 50% 상당액을 가지급보험금으로 지급합니다.# <용어풀이># [가지급보험금]보험금 지급이 늦어지는 경우 '
 '보험수익자 청구에 따라 확정된 보험금을 먼저 지급하는 제도\n'
 '[장해지급률]'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000328',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
