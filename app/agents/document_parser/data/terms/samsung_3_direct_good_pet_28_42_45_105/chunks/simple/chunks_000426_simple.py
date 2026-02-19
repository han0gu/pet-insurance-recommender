from langchain_core.documents import Document

chunk = Document(
    page_content=('① 제21조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따라 특 별약관이 해지되었으나 해약환급금을 받지 않은 '
 '경우(보험계약대출 등에 따라 해약환 급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계 약자는 해지된 '
 '날부터 3년 이내에 회사가 정한 절차에 따라 특별약관의 부활(효력회 복)을 청약할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 74},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000426',
              'chunk_char_len': 201,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
