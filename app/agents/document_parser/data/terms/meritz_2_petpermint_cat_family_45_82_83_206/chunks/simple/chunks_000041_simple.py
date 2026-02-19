from langchain_core.documents import Document

chunk = Document(
    page_content=('. \uf000 회사는 제1항에 따른 만기환급금의 지급시기가 되면 지 급시기 7일 이전에 그 사유와 지급할 금액을 계약자 또는 '
 '보험수익자에게 알려드리며, 만기환급금을 지급함에 있어 지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을 지급할 때의 적립이율 '
 '계산)】에 따릅니다. \uf000 보험료 납입기간 중에 제2조(용어의 정의)에서 정한 적 립보험료를 감액하거나 중도인출을 하는 경우 '
 '제1항의 만기 환급금은 가입시점의 예상금액보다 감소할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 55},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000041',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
