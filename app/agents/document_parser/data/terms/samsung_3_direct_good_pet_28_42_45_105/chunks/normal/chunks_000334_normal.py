from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[음식물]\n'
 '반려견이 일상 생활 중 보호자 또는 생산자의 의도와 상관 없이 섭취할 수 있는 모든 식이 원료와 가공품 및 부산물(뼈, 과일 씨 등 폐기 '
 '대상물질)을 말하며, 사람 및 다른 동물의 식이로 활용될 수 있는 모든 것을 포함합니다. 또한 음식물의 상태(부패, 감염 여부 등)와 '
 '상관없이 모두 포함됩니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 66},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000334',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
