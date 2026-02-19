from langchain_core.documents import Document

chunk = Document(
    page_content=('제22조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '① 제21조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따라 특 별약관이 해지되었으나 해약환급금을 받지 않은 '
 '경우(보험계약대출 등에 따라 해약환 급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포함합니다) 계'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 104},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000618',
              'chunk_char_len': 180,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
