from langchain_core.documents import Document

chunk = Document(
    page_content=('외, 검사비포함)(재가입형) 특별약관을 따르며, 3-1. 반려견 의료비(치과및구강질환포함)(\n'
 '수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사\n'
 '항을 따릅니다. 특별약관 일반사항에서도 정하지 않은 사항은 보통약관을 따릅니다.- 86 -86 / 1813-5. [갱신형] 반려견 '
 '배상책임보장 특별약관# 제 1조 (목적)이 특별약관은 보험계약자(이하「계약자」라 합니다)와 보험회사(이하「회사」라 합니다)\n'
 '사이에 피보험자가 법률상의 배상책임을 부담함으로써 입은 손해에 대한 위험을 보장하'),
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
 'indexing': {'chunk_id': 'chunk_000477',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
