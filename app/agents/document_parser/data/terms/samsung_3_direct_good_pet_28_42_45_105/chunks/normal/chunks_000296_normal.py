from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자는 이 특별약관의 해약환급금 범위 내에서 회사가 정한 방법에 따라 대출(이하 「보험계약대출」이라 합니다)을 받을 수 있습니다. '
 '그러나 순수보장성보험 등 보험상 품의 종류에 따라 보험계약대출이 제한될 수도 있습니다. ② 계약자는 제1항에 따른 보험계약대출금과 그 '
 '이자를 언제든지 상환할 수 있으며 상환 하지 않은 때에는 회사는 보험금, 해약환급금 등의 지급사유가 발생한 날에 지급금에 서 보험계약대출 '
 '원금과 이자를 차감할 수 있습니다'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 58},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000296',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
