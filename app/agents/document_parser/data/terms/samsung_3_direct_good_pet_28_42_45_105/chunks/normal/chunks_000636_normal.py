from langchain_core.documents import Document

chunk = Document(
    page_content=('뒷면에 기재하여 드립니다.\n'
 '1. 지정대리청구인 변경신청서(회사양식) 2. 지정대리청구인의 가족관계등록부(기본증명서 등) 3. 신분증(주민등록증이나 운전면허증 등 '
 '사진이 붙은 정부기관 발행 신분증, 본인이 아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자적 '
 '수단을 할용한 계약자 의사표시의 확인방법 포함)\n'
 '② 제1항에도 불구하고 보험계약에서 지정대리청구인의 지정 기간을 별도로 제한한 경 우, 계약자는 이 특별약관에서도 그 기간에 한하여 '
 '지정대리청구인을 변경 지정할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000636',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
