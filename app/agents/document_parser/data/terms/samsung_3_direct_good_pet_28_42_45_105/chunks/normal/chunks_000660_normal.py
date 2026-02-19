from langchain_core.documents import Document

chunk = Document(
    page_content=('<유의사항>\n'
 '회사는 제2조(특별면책조건의 내용) 제1항 각 호의 질병을 직접적인 원인으로 보험료 납입면제 사 유가 발생한 경우 보험료 납입을 면제하여 '
 '드리지 않습니다.\n'
 '제3조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '회사는 이 특별약관의 부활(효력회복) 청약을 받은 경우에는 계약의 부활(효력회복)을 승\n'
 '낙한 경우에 한하여 보험계약 「보험료의 납입을 연체하여 해지된 계약의 부활(효력회\n'
 '복)」에 따라 이 특별약관의 부활(효력회복)을 취급합니다.\n'
 '제 4조 (준용규정)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 102},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000660',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
