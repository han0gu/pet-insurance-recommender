from langchain_core.documents import Document

chunk = Document(
    page_content=('64 / 181\n'
 '3. 펫 관련 특별약관\n'
 '3-1. 반려견 의료비(치과및구강질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관\n'
 '제 1조 (목적)\n'
 '이 특별약관은 보험계약자(이하「계약자」라 합니다)와 보험회사(이하「회사」라 합니다) 사이에 보험증권에 기재된 반려견의 상해 또는 질병으로 '
 '인한 위험을 보장하기 위하여 체결됩니다.\n'
 '제 2조 (용어의 정의)\n'
 '이 특별약관에서 사용되는 용어의 정의는 이 특별약관의 다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.\n'
 '① 계약 관련 용어'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 66},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000329',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
