from langchain_core.documents import Document

chunk = Document(
    page_content=('- 서 공제계약을 포함합니다.\n'
 '- 피보험자가 의무보험에 가입하여야 함에도 불구하고 가입하지 않은 경우에는 그가 가\n'
 '- 입했더라면 의무보험에서 보상했을 금액을 제1항의 "의무보험에서 보상하는 금액"\n'
 '- 으로 봅니다.\n'
 '# 제6조 (손해의 통지 및 조사)① 계약자 또는 피보험자는 아래와 같은 사실이 있는 경우에는 지체없이 그 내용을 회사\n'
 '에 알려야 합니다.- 1. 사고가 발생하였을 경우 사고가 발생한 때와 곳, 피해자의 주소와 성명, 사고상황\n'
 '- 및 이들 사항의 증인이 있을 경우 그 주소와 성명'),
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
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000491',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
