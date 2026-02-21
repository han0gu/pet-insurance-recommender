from langchain_core.documents import Document

chunk = Document(
    page_content=('- 책임손해」라 합니다)를 이 특별약관에 따라 보상합니다.\n'
 '- ② 제1항에서 회사가 1사고당 보상하는 손해의 범위는 아래와 같습니다.\n'
 '- 1. 피보험자가 피해자에게 지급할 책임을 지는 법률상의 손해배상금\n'
 '- 2. 계약자 또는 피보험자가 지출한 아래의 비용\n'
 '- 가. 피보험자가 제8조(손해방지의무) 제1항 제1호의 손해의 방지 또는 경감을 위하\n'
 '- 여 지출한 필요 또는 유익하였던 비용\n'
 '- 나. 피보험자가 제8조(손해방지의무) 제1항 제2호의 제3자로부터 손해의 배상을'),
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
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 261,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
