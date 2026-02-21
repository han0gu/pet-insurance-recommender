from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 이 법에서 의료기관이라 함은 의료인이 공중 또는 특수 다수인을 위하여 의 료・조산의 업을 행하는 곳을 말합니다. 의료기관의 종별은 '
 '종합병원・병원・ 치과병원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산 원으로 나누어집니다. | 이 법에서 의료기관이라 함은 '
 '의료인이 공중 또는 특수 다수인을 위하여 의 료・조산의 업을 행하는 곳을 말합니다. 의료기관의 종별은 종합병원・병원・ '
 '치과병원・한방병원・요양병원・정신병원・의원・치과의원・한의원 및 조산 원으로 나누어집니다. |'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000345',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
