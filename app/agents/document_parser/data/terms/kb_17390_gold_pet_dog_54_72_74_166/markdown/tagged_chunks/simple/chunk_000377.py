from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '에서 정한 처치 및 수술료, 검사료, 방사선 치료료 등을 포함한 항목에 부여되\n'
 '반\n'
 '는 코드를 말합니다.\n'
 '려\n'
 '동\uf000 제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 병원이나의원 또는 국외의 의료관련법에서 정한 '
 '의료기관에서 발급한 것이어야 합니다. 물\n'
 '관 련 법 규 의료법 제3조(의료기관)이 법에서 의료기관이라 함은 의료인이 공중 또는료・조산의 업을 행하는 곳을 말합니다. 의료기관의 '
 '종별은 종합병원・병원・\n'
 '도\n'
 '치과병원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산 성'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000377',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
