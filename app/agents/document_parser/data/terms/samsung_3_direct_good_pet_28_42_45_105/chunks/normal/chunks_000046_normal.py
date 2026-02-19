from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[지어]\n'
 '2) 1)에 해당하지 않는 경우에는 개인의 사회적 신분에 따르는 위치나 자리를 말함 예) 학생, 미취학아동, 무직 등 [직무] 직책이나 '
 '직업상 책임을 지고 담당하여 맡은 일'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 32},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 108,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
