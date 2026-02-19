from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[심신상실자(心神喪失者)]\n'
 '의식은 있으나 장애의 정도가 심하여 자신의 행위 결과를 합리적으로 판단할 능력을 갖지 못한 사람은 마해 1\n'
 '의사를 결정할 능력이 미약한 사람을 말합니다.\n'
 '3. 계약을 체결할 때 계약에서 정한 피보험자의 나이에 미달되었거나 초과되었을 경 우. 다만, 회사가 나이의 착오를 발견하였을 때 이미 '
 '계약나이에 도달한 경우에는 유효한 계약으로 보나, 제2호의 만 15세 미만자에 관한 예외가 인정되는 것은 아 닙니다.\n'
 '제21조 (계약내용의 변경 등)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 36},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000088',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
