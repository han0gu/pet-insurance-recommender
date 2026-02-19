from langchain_core.documents import Document

chunk = Document(
    page_content=('【자필서명】\n'
 '계약자가 성명기입란에 본인의 성명을 기재하고, 날인란 에 사인(signature) 또는 도장을 찍는 것을 말합니다. 전자서명법 제2조 '
 '제2호에 따른 전자서명을 포함합니다.\n'
 '【 전자서명법 제2조 제2호에 따른 전자서명 】\n'
 '"전자서명"이란 다음 각 목의 사항을 나타내는 데 이용하 기 위하여 전자문서에 첨부되거나 논리적으로 결합된 전 자적 형태의 정보를 '
 '말한다.\n'
 '가. 서명자의 신원 나. 서명자가 해당 전자문서에 서명하였다는 사실'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 70},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000093',
              'chunk_char_len': 245,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
