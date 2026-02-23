from langchain_core.documents import Document

chunk = Document(
    page_content=('자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '습니다.# 제 8조 (보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))부활(효력회복)되는 특별약관의 보장개시는 3-1. 반려견 '
 '의료비(치과및구강질환포함)(수\n'
 '술당일제외, 검사비포함)(재가입형) 특별약관 제22조(보험료의 납입을 연체하여 해지된\n'
 '특별약관의 부활(효력회복))를 따릅니다. 이 경우 부활(효력회복)일을 보험계약일로 하여'),
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
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000424',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
